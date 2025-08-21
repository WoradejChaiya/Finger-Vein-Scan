# train_siamese_cnn.py  — Siamese with PK sampling + batch-hard triplet (fixed finite epochs)
import os, sys, random, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
from PIL import Image
import pandas as pd
from tqdm import tqdm

from models.siamese_cnn_model import SiameseCnnModel
from transforms.transforms_config import transform_pipeline as siamese_tf

# ----------------------- utils -----------------------
def set_seed(seed: int):
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def init_conv1_from_rgb_to_gray(model: SiameseCnnModel):
    try:
        from torchvision.models import resnet18, ResNet18_Weights
        rgb = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        w = rgb.conv1.weight.data.mean(dim=1, keepdim=True)  # [64,1,7,7]
        with torch.no_grad():
            model.backbone.conv1.weight.copy_(w)
        print("[init] conv1 copied from RGB-pretrained → grayscale")
    except Exception as e:
        print(f"[init] conv1 init skipped: {e}")

# ----------------------- dataset (single image) -----------------------
class ImageIDDataset(Dataset):
    def __init__(self, paths, id_indices, transform):
        self.paths = list(paths)
        self.labels = list(id_indices)
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.transform(img), self.labels[i]

# ----------------------- PK BatchSampler (finite) -----------------------
class PKBatchSampler(BatchSampler):
    """
    สุ่มเป็น batch ละ P identities × K images รวม P*K ภาพ
    *finite* — ในหนึ่ง epoch จะ yield ров ров ров เท่ากับ len(self) พอดี
    """
    def __init__(self, labels, P=16, K=4, seed=42, drop_last=True):
        self.labels = list(labels)
        self.P = int(P); self.K = int(K)
        self.drop_last = drop_last
        self.seed = int(seed)
        # group indices by id
        self.idx_by_id = {}
        for i, y in enumerate(self.labels):
            self.idx_by_id.setdefault(y, []).append(i)
        self.ids = list(self.idx_by_id.keys())
        self._epoch = 0

    def __len__(self):
        # จำนวน batch/epoch แบบคร่าว ๆ
        return max(1, len(self.labels) // (self.P * self.K))

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1

        n_batches = len(self)
        for _ in range(n_batches):
            # เลือก P ids (ถ้า id น้อยกว่า P ให้สุ่มซ้ำด้วย choices)
            if len(self.ids) >= self.P:
                chosen_ids = rng.sample(self.ids, self.P)
            else:
                chosen_ids = [rng.choice(self.ids) for _ in range(self.P)]

            batch = []
            for y in chosen_ids:
                pool = self.idx_by_id[y]
                if len(pool) >= self.K:
                    idxs = rng.sample(pool, self.K)          # no replacement
                else:
                    # ถ้ารูปต่อ id น้อย ให้สุ่มแบบซ้ำได้
                    idxs = [rng.choice(pool) for _ in range(self.K)]
                batch.extend(idxs)

            # ปรับขนาดตาม drop_last
            if len(batch) == self.P * self.K or not self.drop_last:
                yield batch

# ----------------------- loss: batch-hard triplet (cosine) -----------------------
def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2):
    emb = F.normalize(emb, p=2, dim=1)
    sim  = emb @ emb.t()         # [B,B]
    dist = 1.0 - sim             # cosine distance
    labels = labels.view(-1, 1)
    pos = labels.eq(labels.t())
    neg = ~pos
    pos.fill_diagonal_(False)

    d_pos = dist.clone(); d_pos[~pos] = -1
    hardest_pos, _ = d_pos.max(dim=1)

    d_neg = dist.clone(); d_neg[~neg] = 10
    hardest_neg, _ = d_neg.min(dim=1)

    valid = hardest_pos > -0.5
    if valid.any():
        return F.relu(hardest_pos[valid] - hardest_neg[valid] + margin).mean()
    return dist.new_tensor(0.0, requires_grad=True)

# ----------------------- main train -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/split_combined.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("results", exist_ok=True)

    # load CSV (train split)
    df = pd.read_csv(args.csv); df = df[df["split"] == "train"].copy()
    df["id"] = df["id"].astype(str)
    paths = df["filepath"].tolist(); ids = df["id"].tolist()
    uniq = sorted(set(ids)); id2idx = {u:i for i,u in enumerate(uniq)}
    y_idx = [id2idx[u] for u in ids]
    with open("results/siamese_cnn_unique_ids.pkl", "wb") as f:
        pickle.dump(uniq, f)

    # dataset / loader
    ds = ImageIDDataset(paths, y_idx, siamese_tf)
    batch_sampler = PKBatchSampler(y_idx, P=args.P, K=args.K, seed=args.seed)
    loader = DataLoader(
        ds, batch_sampler=batch_sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    print(f"[info] steps/epoch ≈ {len(loader)}  (P={args.P}, K={args.K}, PK={args.P*args.K})")

    # model / optim
    model = SiameseCnnModel().to(device)
    init_conv1_from_rgb_to_gray(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train(); running = 0.0
        pbar = tqdm(loader, total=len(loader), ncols=100, desc=f"Epoch {ep}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                z = model.forward_single(x) if hasattr(model, "forward_single") else model.forward_once(x)
                z = F.normalize(z, p=2, dim=1)
                loss = batch_hard_triplet_loss(z, y, margin=args.margin)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        ep_loss = running / max(1, len(loader))
        sch.step()
        print(f"[train] epoch {ep} loss={ep_loss:.5f}")
        if ep_loss < best:
            best = ep_loss
            torch.save(model.state_dict(), "results/siamese_cnn_best.pth")
            print(f"[save] best @ loss={best:.5f}")
    print("done.")

if __name__ == "__main__":
    main()
