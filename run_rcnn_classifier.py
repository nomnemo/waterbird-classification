"""
Leakage-safe? bird species classifier
- Reuses Saahil's crops + metadata_balanced.json
- StratifiedGroupKFold by (species_name_balanced, source_image)
- Choice of backbone: resnet50 (default) or efficientnet_b0
"""

import os, json, math, argparse, random
from collections import Counter

import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------
# Config (override via CLI)
# -----------------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default=r"C:/Users/Audub/saahil_classification/data/metadata_balanced.json")
    ap.add_argument("--crops_root", default=r"C:/Users/Audub/saahil_classification/data/crops")
    ap.add_argument("--save_dir", default=r"C:/Users/Audub/saahil_classification/checkpoints_nomin")
    ap.add_argument("--backbone", choices=["resnet50", "efficientnet_b0"], default="resnet50")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    return ap.parse_args()

# -----------------------
# Dataset
# -----------------------
class BirdDataset(Dataset):
    def __init__(self, records, crops_root, transform=None):
        self.records = records
        self.crops_root = crops_root
        self.transform = transform

        species = sorted({r["species_name_balanced"] for r in records})
        self.species_to_idx = {sp: i for i, sp in enumerate(species)}
        self.idx_to_species = {i: sp for sp, i in self.species_to_idx.items()}

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        rel = r["crop_path"].replace("\\", "/")
        if rel.startswith("crops/"):
            rel = rel[len("crops/"):]
        path = os.path.normpath(os.path.join(self.crops_root, rel))

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        label = self.species_to_idx[r["species_name_balanced"]]
        return img, torch.tensor(label, dtype=torch.long)

# -----------------------
# Transforms
# -----------------------
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.5),
    A.ColorJitter(0.15, 0.15, 0.15, 0.05, p=0.5),
    A.GaussianBlur(p=0.1),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])
eval_tf = A.Compose([
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# -----------------------
# Leakage-safe split
# -----------------------
def stratified_group_split(records, val_ratio, test_ratio, seed=42):
    """Two-stage split with StratifiedGroupKFold by (label, source_image)."""
    labels = [r["species_name_balanced"] for r in records]
    groups = [r["source_image"] for r in records]

    # 1) Train vs Temp (val+test)
    sgkf1 = StratifiedGroupKFold(n_splits=int(1/(val_ratio+test_ratio)), shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(records, labels, groups))
    train_recs = [records[i] for i in train_idx]
    temp_recs  = [records[i] for i in temp_idx]

    # 2) Temp -> Val vs Test (50/50 inside temp)
    temp_labels = [r["species_name_balanced"] for r in temp_recs]
    temp_groups = [r["source_image"] for r in temp_recs]
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    v_idx, t_idx = next(sgkf2.split(temp_recs, temp_labels, temp_groups))
    val_recs = [temp_recs[i] for i in v_idx]
    test_recs= [temp_recs[i] for i in t_idx]
    return train_recs, val_recs, test_recs

# -----------------------
# Models
# -----------------------
def make_model(backbone, num_classes):
    if backbone == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    else:  # efficientnet_b0 (lighter/faster)
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, crit, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = crit(out, y)
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

# -----------------------
# Main
# -----------------------
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.metadata, "r") as f:
        records = json.load(f)

    # Split (grouped by source_image)
    train_recs, val_recs, test_recs = stratified_group_split(
        records, args.val_ratio, args.test_ratio, seed=args.seed
    )
    print(f"Split sizes  -> train: {len(train_recs)} | val: {len(val_recs)} | test: {len(test_recs)}")

    # Datasets / loaders
    train_ds = BirdDataset(train_recs, args.crops_root, transform=train_tf)
    val_ds   = BirdDataset(val_recs,   args.crops_root, transform=eval_tf)
    test_ds  = BirdDataset(test_recs,  args.crops_root, transform=eval_tf)

    num_classes = len(train_ds.species_to_idx)
    print(f"Classes: {num_classes}")
    print("Label map:", train_ds.species_to_idx)

    # Class weights (inverse freq) for imbalance
    cnt = Counter([r["species_name_balanced"] for r in train_recs])
    weights = torch.tensor(
        [1.0 / cnt[sp] for sp in sorted(train_ds.species_to_idx, key=lambda s: train_ds.species_to_idx[s])]
    )
    weights = (weights / weights.sum() * len(weights)).float()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Model / loss / opt / sched
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(args.backbone, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            ckpt = {
                "epoch": epoch,
                "backbone": args.backbone,
                "state_dict": model.state_dict(),
                "val_acc": best_val,
                "label_map": train_ds.species_to_idx
            }
            path = os.path.join(args.save_dir, f"{args.backbone}_best_epoch{epoch}.pth")
            torch.save(ckpt, path)
            print(f"  ✅ Saved best to {path}")

    # Final test evaluation on the best weights (already close; optional reload)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST  | loss {te_loss:.4f} acc {te_acc:.4f}")

if __name__ == "__main__":
    main()
