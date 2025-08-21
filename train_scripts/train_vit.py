# train_scripts/train_vit.py
import os, sys, argparse, random
sys.path.append(os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd

from models.vit_model import ViTModel
from datasets.vit_dataset import ViTDataset
from transforms.vit_transforms import vit_transform_pipeline

# ---------------- Utils ----------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def evaluate(model, loader, criterion, device, use_amp=True):
    model.eval()
    loss_meter, acc_meter, n = 0.0, 0.0, 0
    amp_ctx = torch.cuda.amp.autocast if (use_amp and torch.cuda.is_available()) else torch.cpu.amp.autocast
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx():
                logits = model(imgs)
                loss = criterion(logits, labels)
            bsz = imgs.size(0)
            loss_meter += loss.item() * bsz
            acc_meter  += top1_accuracy(logits, labels) * bsz
            n += bsz
    return loss_meter / max(1, n), acc_meter / max(1, n)

# ----- Early Stopping -----
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -1e9
        self.count = 0
    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric; self.count = 0; return False
        self.count += 1; return self.count > self.patience

# ---------- Data ----------
def _stratified_split_from_train(train_all: pd.DataFrame, frac_val: float, label_col: str, seed: int):
    rng = np.random.default_rng(seed)
    keep_tr_idx, keep_val_idx = [], []
    for cls, grp in train_all.groupby(label_col):
        n = len(grp)
        if n <= 1 or frac_val <= 0:
            keep_tr_idx.extend(grp.index.tolist()); continue
        n_val = max(1, int(round(n * frac_val)))
        n_val = min(n_val, n - 1)  # กันไม่เหลือ train
        val_idx = rng.choice(grp.index.to_numpy(), size=n_val, replace=False)
        tr_idx  = grp.index.difference(val_idx)
        keep_val_idx.extend(val_idx.tolist()); keep_tr_idx.extend(tr_idx.tolist())
    tr_df  = train_all.loc[sorted(keep_tr_idx)].reset_index(drop=True)
    val_df = train_all.loc[sorted(keep_val_idx)].reset_index(drop=True)
    return tr_df, val_df

def build_loaders(csv_path, split_col="split",
                  batch_size=128, num_workers=12, image_size=224,
                  label_col="id", image_col="filepath",
                  val_from_train: float = 0.0, seed: int = 42):
    df = pd.read_csv(csv_path)

    if val_from_train and val_from_train > 0:
        base = df[df[split_col] == "train"].reset_index(drop=True)
        train_df, val_df = _stratified_split_from_train(base, val_from_train, label_col, seed)
        print(f"[split] val_from_train={val_from_train:.2f} -> train={len(train_df)} val={len(val_df)} (จาก train เดิม {len(base)})")
    else:
        train_df = df[df[split_col] == "train"].reset_index(drop=True)
        val_df   = df[df[split_col] == "val"].reset_index(drop=True)

    # map labels -> 0..C-1 โดยอิง "คลาสใน train"
    train_classes = sorted(train_df[label_col].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    num_classes  = len(train_classes)

    train_labels = train_df[label_col].map(class_to_idx).astype(int).to_numpy()

    # ถ้า val มีคลาสที่ไม่อยู่ใน train ให้ตัดทิ้ง (กัน val_acc = 0 แบบที่เจอ)
    val_labels_mapped = val_df[label_col].map(class_to_idx)
    mask = val_labels_mapped.notna()
    dropped = int((~mask).sum())
    if dropped > 0:
        print(f"[warn] ตัดตัวอย่าง val {dropped} รายการที่ label ไม่อยู่ใน train ออก")
    val_df = val_df[mask].reset_index(drop=True)
    val_labels = val_labels_mapped[mask].astype(int).to_numpy()

    train_tf = vit_transform_pipeline(image_size=image_size, train=True)
    val_tf   = vit_transform_pipeline(image_size=image_size, train=False)

    train_set = ViTDataset(train_df, labels=train_labels, transform=train_tf, image_col=image_col)
    val_set   = ViTDataset(val_df,   labels=val_labels,   transform=val_tf,   image_col=image_col)

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers and num_workers > 0:
        dl_kwargs.update(dict(prefetch_factor=4, persistent_workers=True))

    train_loader = DataLoader(train_set, shuffle=True,  drop_last=True, **dl_kwargs)
    val_loader   = DataLoader(val_set,   shuffle=False, drop_last=False, **dl_kwargs)

    print(f"[labels] classes={num_classes} | train range=({train_labels.min()}..{train_labels.max()}) | "
          f"val range=({val_labels.min()}..{val_labels.max()}) | val size={len(val_set)}")
    return train_loader, val_loader, num_classes, class_to_idx

def save_ckpt(path, model, optimizer, epoch, best_val_acc, class_to_idx):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "meta": {"class_to_idx": class_to_idx},
    }, path)

# ---------------- Train loops ----------------
def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0, use_amp=True):
    model.train()
    running = 0.0; n = 0
    scaler = train_one_epoch.__dict__.setdefault("_scaler", torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available()))
    for imgs, labels in tqdm(loader, leave=False, desc="train"):
        imgs = imgs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                logits = model(imgs); loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs); loss = criterion(logits, labels)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        running += loss.item() * imgs.size(0); n += imgs.size(0)
    return running / max(1, n)

def make_optimizer_stageA(model, lr_head=5e-4, weight_decay=0.05):
    params = [p for p in model.classifier.parameters() if p.requires_grad]
    return AdamW(params, lr=lr_head, weight_decay=weight_decay)

def make_optimizer_stageB(model, lr_head=5e-4, lr_backbone=5e-5, weight_decay=0.05):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    return AdamW([
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params,     "lr": lr_head,     "weight_decay": weight_decay},
    ])

def make_scheduler_with_warmup(optimizer, total_epochs, warmup_epochs=5, eta_min=1e-5):
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=eta_min)
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = cosine
    return scheduler

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/split_combined.csv")
    parser.add_argument("--model_name", type=str, default="vit_b_16")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    # สเตจ (รวม 50 epochs)
    parser.add_argument("--epochs_A", type=int, default=5)
    parser.add_argument("--epochs_B", type=int, default=45)
    parser.add_argument("--warmup_A", type=int, default=3)
    parser.add_argument("--warmup_B", type=int, default=8)

    # LR/WD
    parser.add_argument("--lr_head_A", type=float, default=5e-4)
    parser.add_argument("--lr_head_B", type=float, default=5e-4)
    parser.add_argument("--lr_backbone_B", type=float, default=5e-5)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "cls"])
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ckpt", type=str, default="results/checkpoints/vit_best.pt")

    # Early stopping
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)

    # AMP
    parser.add_argument("--amp", action="store_true", help="เปิด Automatic Mixed Precision")

    # NEW: split val จาก train (fraction 0..1); ถ้า 0 จะใช้คอลัมน์ 'val' ใน CSV
    parser.add_argument("--val_from_train", type=float, default=0.1)

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, num_classes, class_to_idx = build_loaders(
        args.csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.img_size,
        label_col="id",
        image_col="filepath",
        val_from_train=args.val_from_train,
        seed=args.seed
    )

    # Model
    model = ViTModel(
        model_name=args.model_name,
        pretrained=True,
        num_classes=num_classes,
        pool=args.pool,
        drop_path=args.drop_path
    ).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = -1.0

    # ===== Stage A: head-only =====
    model.freeze_backbone()
    optimizer = make_optimizer_stageA(model, lr_head=args.lr_head_A, weight_decay=args.weight_decay)
    scheduler = make_scheduler_with_warmup(optimizer, total_epochs=args.epochs_A,
                                           warmup_epochs=args.warmup_A, eta_min=args.eta_min)
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    for epoch in range(args.epochs_A):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     grad_clip=args.grad_clip, use_amp=args.amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp=args.amp)
        scheduler.step()
        print(f"[A][{epoch+1}/{args.epochs_A}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_ckpt(args.ckpt, model, optimizer, epoch, best_val_acc, class_to_idx)
        if stopper.step(val_acc):
            print(f"[A] Early stopped (patience={args.patience})")
            break

    # ===== Stage B: unfreeze last 4 blocks =====
    model.unfreeze_last_blocks(n_blocks=4)
    optimizer = make_optimizer_stageB(model, lr_head=args.lr_head_B, lr_backbone=args.lr_backbone_B,
                                      weight_decay=args.weight_decay)
    scheduler = make_scheduler_with_warmup(optimizer, total_epochs=args.epochs_B,
                                           warmup_epochs=args.warmup_B, eta_min=args.eta_min)
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta)

    for epoch in range(args.epochs_B):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                     grad_clip=args.grad_clip, use_amp=args.amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp=args.amp)
        scheduler.step()
        print(f"[B][{epoch+1}/{args.epochs_B}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_ckpt(args.ckpt, model, optimizer, epoch, best_val_acc, class_to_idx)
        if stopper.step(val_acc):
            print(f"[B] Early stopped (patience={args.patience})")
            break

    print(f"Done. Best val_top1 = {best_val_acc:.4f}. Checkpoint -> {args.ckpt}")

if __name__ == "__main__":
    main()
