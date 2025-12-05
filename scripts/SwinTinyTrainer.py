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
from scripts.DataLoader import set_up_data_loaders

# ===== config =====
MODEL_NAME   = "swin_tiny_patch4_window7_224"
EPOCHS       = 40 
LR           = 3e-4
WEIGHT_DECAY = 0.01
ACCUM_STEPS  = 1 
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu" # what is this doing?

AMP          = True # what is thos
CKPT_PATH    = "best_swin.pt"
MAX_PER_CLASS = 100 
OUT_DIR      = Path("runs_swin2")  
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

def plot_curves(history, path):
    """
    Plot training vs validation loss and accuracy.

    Produces a single image with two subplots:
      - Top:  train_loss (blue) and val_loss (red)
      - Bottom: train_acc (blue) and val_acc (red)
    """
    ep = np.arange(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)

    # Loss subplot
    ax_loss.plot(ep, history["train_loss"], label="train_loss", color="blue")
    ax_loss.plot(ep, history["val_loss"], label="val_loss", color="red")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()

    # Accuracy subplot
    ax_acc.plot(ep, history["train_acc"], label="train_acc", color="blue")
    ax_acc.plot(ep, history["val_acc"], label="val_acc", color="red")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_two_cms(y1, p1, y2, p2, classes, path, titles=("Validation", "Test")):
    """Plot two row-normalized confusion matrices side-by-side and save to path.

    Values are expressed as percentages per true class (rows sum to 100%)."""
    from sklearn.metrics import confusion_matrix
    labels = list(range(len(classes)))

    cm1 = confusion_matrix(y1, p1, labels=labels)
    cm2 = confusion_matrix(y2, p2, labels=labels)

    # Row-normalize to percentages (per true class).
    def _row_normalize(cm):
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = np.divide(cm, row_sums, where=row_sums > 0) * 100.0
        pct[np.isnan(pct)] = 0.0
        return pct

    cm1_pct = _row_normalize(cm1)
    cm2_pct = _row_normalize(cm2)

    # Use constrained_layout so the colorbar and subplots do not overlap.
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    vmin, vmax = 0.0, 100.0
    im = None
    for ax, cm_pct, title in zip(axes, (cm1_pct, cm2_pct), titles):
        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ticks = np.arange(len(classes))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
        # annotate with percentages
        thresh = (cm_pct.max() + cm_pct.min()) / 2.0 if cm_pct.size else 0.0
        for i in range(cm_pct.shape[0]):
            for j in range(cm_pct.shape[1]):
                val = cm_pct[i, j]
                text = f"{val:.1f}" if val > 0 else "0.0"
                color = "white" if val > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=6)

    # Place a single colorbar to the right of both subplots.
    cbar = fig.colorbar(im, ax=axes, location="right", fraction=0.046, pad=0.02)
    cbar.set_label("Percentage (%)")
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
    """Full report + mAP, returns metrics and predictions (no file output)."""
    y_true, y_pred, probs = eval_collect(model, dl, len(classes))

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

    metrics = {
        "macro_f1": float(macro_f1),
        "map_macro": float(mAP_macro),
        "n_samples": int(len(y_true)),
    }
    # return metrics and the raw y/pred so the caller can compose combined plots
    return metrics, y_true, y_pred

def main():
    global OUT_DIR

    # ----- create run-specific output directory -----
    run_name = make_run_dir_name(MODEL_NAME, MAX_PER_CLASS, EPOCHS, LR, WEIGHT_DECAY, ACCUM_STEPS)
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    OUT_DIR = run_dir # Update global OUT_DIR to this run's subdirectory so all outputs go there
    print(f"[info] run directory: {OUT_DIR}")


    # ----- set up data loaders -----
    dl_train, dl_val, dl_test, meta = set_up_data_loaders(max_per_class=MAX_PER_CLASS)
    classes = meta["classes"]; num_classes = len(classes)
    # spit out validation and test split distributions
    comp_val  = split_composition(dl_val.dataset,  classes)
    comp_test = split_composition(dl_test.dataset, classes)
    with open(OUT_DIR / "split_composition.json", "w", encoding="utf-8") as f:
        json.dump({"val": comp_val, "test": comp_test}, f, indent=2)
    print("[info] saved split_composition.json")
    
    
    # ----- model / opt -----
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(DEVICE)
    opt   = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler= GradScaler(device="cuda", enabled=AMP)
    # Print out the model config used for this run
    print(f"[info] model: {MODEL_NAME}")
    print(f"[info] epochs: {EPOCHS}, lr: {LR}, weight_decay: {WEIGHT_DECAY}, accum_steps: {ACCUM_STEPS}")

    # training logs
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val = 0.0

    # ----- training loop -----
    for ep in range(1, EPOCHS+1):
        # train pass
        
        # put model in training mode
        model.train() 
        
        # set up running metrics
        running_loss = running_correct = running_count = 0

        # clear gradients at start of epoch to avoid accumulation from previous epoch ? 
        opt.zero_grad(set_to_none=True)
        
        # iterate over training batches
        for step, (xb, yb) in enumerate(dl_train, start=1):
            # move batch to device
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            
            # forward pass with mixed precision
            with autocast(device_type="cuda", enabled=AMP):
                # compute logits and loss
                # logits is (batch_size, num_classes) tensor of raw class scores after model forward pass
                logits = model(xb) 
                
                # use cross-entropy loss for multi-class classification
                loss   = nn.functional.cross_entropy(logits, yb)

            # backward pass with gradient scaling for AMP
            scaler.scale(loss / ACCUM_STEPS).backward()
            if step % ACCUM_STEPS == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

            # update running metrics
            with torch.no_grad():
                pred = logits.argmax(1)
                running_correct += (pred == yb).sum().item()
                running_count   += yb.size(0)
                running_loss    += loss.item() * yb.size(0)

        sched.step()
        train_acc  = running_correct / max(1, running_count)
        train_loss = running_loss  / max(1, running_count)

        # validation pass
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
    print("[info] saved curves.png")

    # ----- final evaluation -----
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    val_summary, val_y, val_p = evaluate_full(model, dl_val,  classes, header="Validation report", save_prefix="val")
    test_summary, test_y, test_p = evaluate_full(model, dl_test, classes, header="Test report",       save_prefix="test")

    # combined confusion matrices side-by-side
    plot_two_cms(val_y, val_p, test_y, test_p, classes, OUT_DIR / "val_test_cms.png", titles=("Validation", "Test"))

    print("[info] saved val_test_cms.png")

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
