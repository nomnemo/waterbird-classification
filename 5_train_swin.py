import time
from datetime import timedelta
from pathlib import Path
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch, timm
from torch import nn
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             average_precision_score)
from _4_dataloader import build_loaders

# ===== config =====
MODEL_NAME   = "swin_tiny_patch4_window7_224"
EPOCHS       = 10 # 10,20,30
LR           = 3e-4 # 0.0003, 0.0001, 0.00005
WEIGHT_DECAY = 0.01 # 0.01, 0.03
ACCUM_STEPS  = 1 # (gradient accumulation steps) 
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
AMP          = True
CKPT_PATH    = "best_swin.pt"
MAX_PER_CLASS = 100 # tried 50, 100, 200, 500.
OUT_DIR      = Path("runs_swin")  # logs/artifacts live here
OUT_DIR.mkdir(exist_ok=True)
# ==================

def make_run_dir_name(model_name: str, max_per_class: int, epochs: int, lr: float, 
                       weight_decay: float, accum_steps: int) -> str:
    """
    Format a descriptive run directory name from hyperparameters.
    Example: "swin_mpc50_ep20_lr0003_wd0050_as1"
    
    Args:
        model_name: base model identifier (e.g., "swin_tiny_patch4_window7_224")
        max_per_class: max samples per class
        epochs: number of epochs
        lr: learning rate (formatted as integer code)
        weight_decay: weight decay (formatted as integer code)
        accum_steps: gradient accumulation steps
    
    Returns:
        descriptive string for use as subdirectory name
    """
    # shorten model name
    if "swin" in model_name.lower():
        model_short = "swin"
    else:
        model_short = model_name[:8]
    
    # format LR: 3e-4 = 0.0003 → scale by 1e6 → 300 → "0300"
    lr_int = int(round(lr * 1e6))
    lr_str = f"{lr_int:04d}"
    
    # format weight decay: 0.05 → scale by 10000 → 500 → "0500"
    wd_int = int(round(weight_decay * 10000))
    wd_str = f"{wd_int:04d}"
    
    run_name = f"{model_short}_mpc{max_per_class}_ep{epochs}_lr{lr_str}_wd{wd_str}_as{accum_steps}"
    return run_name

def fmt(s: float) -> str:
    return str(timedelta(seconds=s))

def plot_curves(history, path):
    fig = plt.figure(figsize=(7.5,4.5))
    ep = np.arange(1, len(history["train_loss"])+1)
    plt.plot(ep, history["train_loss"], label="train_loss")
    plt.plot(ep, history["val_loss"],   label="val_loss")
    plt.plot(ep, history["train_acc"],  label="train_acc")
    plt.plot(ep, history["val_acc"],    label="val_acc")
    plt.xlabel("epoch"); plt.legend(); plt.tight_layout()
    fig.savefig(path, dpi=180); plt.close(fig)

def plot_cm(y_true, y_pred, classes, path="cm.png"):
    from sklearn.metrics import confusion_matrix
    labels = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_two_cms(y1, p1, y2, p2, classes, path, titles=("Validation", "Test")):
    """Plot two confusion matrices side-by-side and save to path."""
    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(y1, p1, labels=list(range(len(classes))))
    cm2 = confusion_matrix(y2, p2, labels=list(range(len(classes))))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, cm, title in zip(axes, (cm1, cm2), titles):
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        # annotate with counts
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm[i, j])
                color = "white" if val > thresh else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=6)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def split_composition(ds, classes):
    # ds is BirdDataset with .rows that include 'species_name'
    from collections import Counter
    cnt = Counter([r["species_name"] for r in ds.rows])
    # ensure every class shows up (even if 0)
    return {c:int(cnt.get(c,0)) for c in classes}

def eval_collect(model, dl, num_classes):
    """Return (y_true_list, y_pred_list, y_proba_array)"""
    model.eval()
    y_true, y_pred = [], []
    probs = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            with autocast(device_type="cuda", enabled=AMP):
                logits = model(xb)
                p = torch.softmax(logits, dim=1)
            probs.append(p.cpu().numpy())
            y_true.extend(yb.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, num_classes), dtype=np.float32)
    return y_true, y_pred, probs

def compute_map_ovr(y_true, probs, num_classes):
    """One-vs-rest AP per class, macro mAP."""
    if len(y_true) == 0:
        return 0.0, np.zeros(num_classes)
    y_true = np.array(y_true)
    Y = np.zeros((len(y_true), num_classes), dtype=np.int32)
    Y[np.arange(len(y_true)), y_true] = 1
    ap_per_class = []
    for k in range(num_classes):
        try:
            ap = average_precision_score(Y[:,k], probs[:,k])
        except ValueError:
            ap = 0.0
        ap_per_class.append(ap if np.isfinite(ap) else 0.0)
    return float(np.mean(ap_per_class)), np.array(ap_per_class)

def evaluate_full(model, dl, classes, header, save_prefix):
    """Full report + mAP + CM + saves predictions/probs."""
    t0 = time.perf_counter()
    y_true, y_pred, probs = eval_collect(model, dl, len(classes))
    t_forward = time.perf_counter() - t0

    labels = list(range(len(classes)))
    print(f"\n{header}:")
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=classes,
        digits=3,
        zero_division=0
    ))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    print(f"{header} macro-F1: {macro_f1:.3f}")

    mAP_macro, ap_cls = compute_map_ovr(y_true, probs, len(classes))
    print(f"{header} mAP (macro, one-vs-rest): {mAP_macro:.3f}")

    # save per-class AP (no longer saving raw arrays)
    with open(OUT_DIR / f"{save_prefix}_ap_per_class.json", "w", encoding="utf-8") as f:
        json.dump({cls: float(ap) for cls, ap in zip(classes, ap_cls.tolist())}, f, indent=2)

    plot_cm(y_true, y_pred, classes, OUT_DIR / f"{save_prefix}_cm.png")

    metrics = {
        "macro_f1": float(macro_f1),
        "map_macro": float(mAP_macro),
        "n_samples": int(len(y_true)),
    }
    # return metrics and the raw y/pred so the caller can compose combined plots
    return metrics, y_true, y_pred

def main():
    global OUT_DIR
    script_start = time.perf_counter()

    # ----- create run-specific output directory -----
    run_name = make_run_dir_name(MODEL_NAME, MAX_PER_CLASS, EPOCHS, LR, WEIGHT_DECAY, ACCUM_STEPS)
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Update global OUT_DIR to this run's subdirectory so all outputs go there
    OUT_DIR = run_dir
    print(f"[info] run directory: {OUT_DIR}")

    # ----- loaders -----
    t0 = time.perf_counter()

    dl_train, dl_val, dl_test, meta = build_loaders(max_per_class=MAX_PER_CLASS)
    t_build = time.perf_counter() - t0

    classes = meta["classes"]; num_classes = len(classes)

    # Print out the config used for this run
    print(f"[info] model: {MODEL_NAME}")
    print(f"[info] epochs: {EPOCHS}, lr: {LR}, weight_decay: {WEIGHT_DECAY}, accum_steps: {ACCUM_STEPS}")

    # split composition
    comp_val  = split_composition(dl_val.dataset,  classes)
    comp_test = split_composition(dl_test.dataset, classes)
    with open(OUT_DIR / "split_composition.json", "w", encoding="utf-8") as f:
        json.dump({"val": comp_val, "test": comp_test}, f, indent=2)
    print("[info] saved split_composition.json")

    # ----- model / opt -----
    t1 = time.perf_counter()
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(DEVICE)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler= GradScaler(device="cuda", enabled=AMP)
    t_model = time.perf_counter() - t1

    # training logs
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0

    # ----- training -----
    for ep in range(1, EPOCHS+1):
        t_ep = time.perf_counter()
        model.train()
        running_loss = running_correct = running_count = 0

        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(dl_train, start=1):
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            with autocast(device_type="cuda", enabled=AMP):
                logits = model(xb)
                loss   = nn.functional.cross_entropy(logits, yb)

            scaler.scale(loss / ACCUM_STEPS).backward()
            if step % ACCUM_STEPS == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            with torch.no_grad():
                pred = logits.argmax(1)
                running_correct += (pred == yb).sum().item()
                running_count   += yb.size(0)
                running_loss    += loss.item() * yb.size(0)

        sched.step()
        train_acc  = running_correct / max(1, running_count)
        train_loss = running_loss  / max(1, running_count)

        # quick val
        t_val = time.perf_counter()
        model.eval()
        v_loss = v_correct = v_count = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with autocast(device_type="cuda", enabled=AMP):
                    logits = model(xb)
                    loss   = nn.functional.cross_entropy(logits, yb)
                v_loss    += loss.item() * yb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_count   += yb.size(0)
        val_acc  = v_correct / max(1, v_count)
        val_loss = v_loss    / max(1, v_count)
        t_ep_val = time.perf_counter() - t_val

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"ep {ep:02d} | train acc {train_acc:.3f} loss {train_loss:.3f} "
            f"| val acc {val_acc:.3f} loss {val_loss:.3f} "
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "name": MODEL_NAME}, CKPT_PATH)

    # save curves & csv
    plot_curves(history, OUT_DIR / "curves.png")
    import csv
    with open(OUT_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])
        for i in range(EPOCHS):
            w.writerow([i+1, history["train_loss"][i], history["train_acc"][i],
                        history["val_loss"][i], history["val_acc"][i]])
    with open(OUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "accum_steps": ACCUM_STEPS
        }, f, indent=2)
    print("[info] saved curves.png, metrics.csv, run_config.json")

    # ----- final evaluation -----
    t_load = time.perf_counter()
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    val_summary, val_y, val_p = evaluate_full(model, dl_val,  classes, header="Validation report", save_prefix="val")
    test_summary, test_y, test_p = evaluate_full(model, dl_test, classes, header="Test report",       save_prefix="test")

    # combined confusion matrices side-by-side
    plot_two_cms(val_y, val_p, test_y, test_p, classes, OUT_DIR / "val_test_cms.png", titles=("Validation", "Test"))

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_summary, "test": test_summary}, f, indent=2)
    print("[info] saved cm_val.png, cm_test.png, val_test_cms.png, *_ap_per_class.json, summary.json")

    # ----- total time -----
    total_dt = time.perf_counter() - script_start
    print(f"[time] TOTAL script wall time: {fmt(total_dt)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin with configurable hyperparameters")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH)
    parser.add_argument("--device", type=str, default=DEVICE)
    args = parser.parse_args()

    # override module-level defaults with CLI values
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    MAX_PER_CLASS = args.max_per_class
    ACCUM_STEPS = args.accum_steps
    MODEL_NAME = args.model_name
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_PATH = args.ckpt_path
    DEVICE = args.device

    main()
