# evaluate_models.py
import os, sys
sys.path.append(os.path.abspath("."))

import argparse, random, pickle
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from models.siamese_cnn_model import SiameseCnnModel
from models.vit_model import ViTModel
from models.efficientnet_b0_model import EfficientNetB0Model

def set_seed(seed: int):
    import numpy as _np
    random.seed(seed); _np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def resolve_image_path(filepath: str, data_root: str = "data") -> str:
    if isinstance(filepath, str) and os.path.exists(filepath): return filepath
    fname = os.path.basename(str(filepath))
    for root, _, files in os.walk(data_root):
        if fname in files: return os.path.join(root, fname)
    return None

class CsvImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.paths = df["filepath"].tolist()
        self.ids   = df["id"].astype(str).tolist()
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img), self.ids[idx]

def default_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([T.Grayscale(1), T.Resize((image_size, image_size)),
                      T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

def try_import_project_transforms():
    siamese_tf = vit_tf = eff_tf = None
    try: from transforms.transforms_config import transform_pipeline as siamese_tf
    except Exception: pass
    try: from transforms.vit_transforms import vit_transform_pipeline as vit_tf
    except Exception: pass
    try: from transforms.efficientnet_b0_transforms import transform_pipeline as eff_tf
    except Exception: pass
    return siamese_tf, vit_tf, eff_tf

def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True); return x / (n + eps)

def cosine_scores(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return l2_normalize_np(A) @ l2_normalize_np(B).T

def far_frr_eer_from_scores(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score); y_true_sorted = y_true[order]; scores_sorted = y_score[order]
    P = int(np.sum(y_true_sorted == 1)); N = int(np.sum(y_true_sorted == 0))
    if P == 0 or N == 0: raise RuntimeError("ไม่สามารถคำนวณ EER ได้ (ไม่มีคู่ genuine หรือ impostor)")
    tp = fp = 0; best = (1.0, 1.0, 1.0, 1.0, 0.0)  # diff, eer, far, frr, thr
    for i in range(len(scores_sorted)):
        if y_true_sorted[i] == 1: tp += 1
        else: fp += 1
        far = fp / N; frr = (P - tp) / P; diff = abs(far - frr)
        if diff < best[0]:
            eer = (far + frr) / 2.0; best = (diff, eer, far, frr, scores_sorted[i])
    _, eer, far, frr, thr = best; return float(eer), float(far), float(frr), float(thr)

def confusion_matrix_2x2_from_scores(y_true: np.ndarray, y_score: np.ndarray, thr: float):
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp

def topk_identification(g_feats: np.ndarray, g_ids: List[str], p_feats: np.ndarray, p_ids: List[str], k: int = 1):
    S = cosine_scores(p_feats, g_feats)
    topk_idx = np.argpartition(-S, kth=min(k, S.shape[1]-1), axis=1)[:, :k]
    correct, y_pred = 0, []
    for i in range(S.shape[0]):
        best_j = topk_idx[i, np.argmax(S[i, topk_idx[i]])]
        y_pred.append(g_ids[best_j])
        if g_ids[best_j] == p_ids[i]: correct += 1
    top1 = correct / len(p_ids) if len(p_ids) > 0 else float("nan")
    return top1, y_pred

def build_verification_pairs(g_feats: np.ndarray, g_ids: List[str], p_feats: np.ndarray, p_ids: List[str],
                             neg_per_pos: int = 1, seed: int = 42):
    rng = np.random.default_rng(seed)
    S = cosine_scores(p_feats, g_feats); g_index_by_id = {gid: j for j, gid in enumerate(g_ids)}
    gallery_indices = np.arange(len(g_ids)); g_ids_arr = np.array(g_ids)
    imp_idx_per_id = {gid: gallery_indices[g_ids_arr != gid] for gid in g_ids}
    y_true, y_score = [], []
    for i, pid in enumerate(p_ids):
        if pid not in g_index_by_id: continue
        gj = g_index_by_id[pid]; y_true.append(1); y_score.append(float(S[i, gj]))
        imp_pool = imp_idx_per_id[pid]
        if len(imp_pool) == 0: continue
        choose = min(neg_per_pos, len(imp_pool))
        for j in rng.choice(imp_pool, size=choose, replace=False):
            y_true.append(0); y_score.append(float(S[i, j]))
    return np.array(y_true, int), np.array(y_score, float)

def load_csv_checked(path: str, data_root: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"filepath", "id"}.issubset(df.columns):
        raise RuntimeError(f"CSV {path} ต้องมีคอลัมน์ filepath และ id")
    df["id"] = df["id"].astype(str)
    df["filepath"] = df["filepath"].apply(lambda p: resolve_image_path(p, data_root))
    df = df[df["filepath"].notnull()].reset_index(drop=True)
    if len(df) == 0: raise RuntimeError(f"CSV {path} ว่างเปล่าหลังแก้ path")
    return df

def extract_embeddings(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval(); feats = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            if hasattr(model, "extract_embeddings"):
                z = model.extract_embeddings(imgs, l2norm=True)
            elif hasattr(model, "forward_single"):
                z = F.normalize(model.forward_single(imgs), p=2, dim=1)
            elif hasattr(model, "extract_features"):
                z = F.normalize(model.extract_features(imgs), p=2, dim=1)
            elif hasattr(model, "forward_features"):
                z = F.normalize(model.forward_features(imgs), p=2, dim=1)
            else:
                raise RuntimeError("โมเดลไม่มีเมธอด extract_embeddings/forward_single/extract_features/forward_features")
            feats.append(z.detach().cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 128), np.float32)

def load_forgiving(model, state_dict, ignore_prefixes=()):
    msd = model.state_dict(); keep, dropped = {}, []
    for k, v in state_dict.items():
        if any(k.startswith(p) for p in ignore_prefixes): dropped.append((k, "ignored-prefix")); continue
        if (k in msd) and (msd[k].shape == v.shape): keep[k] = v
        else: dropped.append((k, "shape/unexpected"))
    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"[load] kept {len(keep)} keys, dropped {len(dropped)}; missing={len(missing)}, unexpected={len(unexpected)}")
    return missing, unexpected, dropped

def sanity(name: str, g_feats: np.ndarray, p_feats: np.ndarray, g_ids: List[str], p_ids: List[str], k: int = 1000):
    if len(p_feats) == 0 or len(g_feats) == 0: print(f"[sanity/{name}] (no feats)"); return
    k = min(k, len(p_feats)); S = cosine_scores(p_feats[:k], g_feats); g_index = {gid: j for j, gid in enumerate(g_ids)}
    g_sc, i_sc = [], []
    for i in range(k):
        pid = p_ids[i]
        if pid in g_index:
            j = g_index[pid]; g_sc.append(S[i, j])
            mask = np.ones(S.shape[1], dtype=bool); mask[j] = False
            i_sc.append(S[i, mask].mean())
    if len(g_sc) == 0: print(f"[sanity/{name}] no closed-set pairs to analyze"); return
    g_mean, i_mean = float(np.mean(g_sc)), float(np.mean(i_sc))
    print(f"[sanity/{name}] mean cosine: genuine={g_mean:.4f} impostor={i_mean:.4f} (Δ={g_mean - i_mean:.4f})")

def dbg_unique(name: str, F: np.ndarray):
    if F.size == 0: return
    uniq = np.unique(np.round(F, 3), axis=0).shape[0]
    print(f"[dbg/{name}] shape={F.shape}, unique_rows@1e-3={uniq}, std={F.std():.6f}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate Siamese/ViT/EfficientNet on Verification + Identification")
    ap.add_argument("--gallery-csv", type=str, default="data/test_gallery.csv")
    ap.add_argument("--probe-csv",   type=str, default="data/test_probe.csv")
    ap.add_argument("--data-root",   type=str, default="data")
    ap.add_argument("--save-dir",    type=str, default="results")
    ap.add_argument("--batch-size",  type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--image-size",  type=int, default=224)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--w-siamese",   type=str, default="results/siamese_cnn_best.pth")
    ap.add_argument("--w-vit",       type=str, default="results/checkpoints/vit_best.pt")
    ap.add_argument("--w-eff",       type=str, default="results/efficientnet_b0_best.pth")
    ap.add_argument("--print-uniq",  action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # CSV
    df_g = load_csv_checked(args.gallery_csv, args.data_root)
    df_p = load_csv_checked(args.probe_csv, args.data_root)
    g_ids = df_g["id"].astype(str).tolist(); p_ids = df_p["id"].astype(str).tolist()
    if not set(p_ids).issubset(set(g_ids)):
        missing = sorted(list(set(p_ids) - set(g_ids)))[:10]
        raise RuntimeError(f"probe มี ID ที่ไม่มีใน gallery จำนวน {len(set(p_ids)-set(g_ids))} เช่น {missing} ...")

     # -------- Transforms --------
    siamese_tf, vit_tf, eff_tf = try_import_project_transforms()

    def ensure_compose(tf, *, three_ch: bool):
        """รับได้ทั้ง None / Compose / ฟังก์ชัน -> คืนค่าเป็น Compose เสมอ"""
        if tf is None:
            if three_ch:
                return T.Compose([
                    T.Grayscale(num_output_channels=3),
                    T.Resize((args.image_size, args.image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                return default_transform(args.image_size)

        if callable(tf):  # เป็นฟังก์ชัน -> ลองเรียกให้กลายเป็น Compose
            for sig in [
                dict(image_size=args.image_size, train=False),
                dict(image_size=args.image_size),
                dict(),
            ]:
                try:
                    return tf(**sig)
                except TypeError:
                    continue
            # ถ้าเรียกไม่ได้จริง ๆ ก็ถือว่าเป็น Compose อยู่แล้ว
        return tf

    siamese_tf = ensure_compose(siamese_tf, three_ch=False)  # 1-channel
    vit_tf     = ensure_compose(vit_tf,     three_ch=True)   # 3-channel
    eff_tf     = ensure_compose(eff_tf,     three_ch=True)   # 3-channel

    print("[TF] siamese =", siamese_tf)
    print("[TF] vit     =", vit_tf)
    print("[TF] eff     =", eff_tf)

    # Loaders
    pin_mem = (device.type == "cuda")
    dl_kwargs = dict(batch_size=args.batch_size, shuffle=False,
                     num_workers=args.num_workers, pin_memory=pin_mem)
    if args.num_workers and args.num_workers > 0:
        dl_kwargs.update(dict(prefetch_factor=4, persistent_workers=True))
    dl_g_siam = DataLoader(CsvImageDataset(df_g, siamese_tf), **dl_kwargs)
    dl_p_siam = DataLoader(CsvImageDataset(df_p, siamese_tf), **dl_kwargs)
    dl_g_vit  = DataLoader(CsvImageDataset(df_g, vit_tf), **dl_kwargs)
    dl_p_vit  = DataLoader(CsvImageDataset(df_p, vit_tf), **dl_kwargs)
    dl_g_eff  = DataLoader(CsvImageDataset(df_g, eff_tf), **dl_kwargs)
    dl_p_eff  = DataLoader(CsvImageDataset(df_p, eff_tf), **dl_kwargs)

    # Siamese
    siamese = SiameseCnnModel().to(device)
    if os.path.exists(args.w_siamese):
        sd = torch.load(args.w_siamese, map_location=device, weights_only=False)
        load_forgiving(siamese, sd)
    else:
        print(f"[Warn] ไม่พบ weights Siamese ที่ {args.w_siamese}")

    # ViT (embedding ตรง)
    vit = ViTModel(num_classes=0, pool="mean").to(device)
    if os.path.exists(args.w_vit):
        sd_raw = torch.load(args.w_vit, map_location=device, weights_only=False)
        sd = sd_raw.get("model", sd_raw)
        load_forgiving(vit, sd, ignore_prefixes=("classifier.",))
    else:
        print(f"[Warn] ไม่พบ weights ViT ที่ {args.w_vit}")

    # EfficientNet-B0
    eff = EfficientNetB0Model(num_classes=1000).to(device)
    if os.path.exists(args.w_eff):
        sd = torch.load(args.w_eff, map_location=device, weights_only=False)
        if not any(k.startswith("embed_fc.") for k in sd.keys()) and hasattr(eff, "embed_fc"):
            eff.embed_fc = nn.Identity(); print("[EffB0] ไม่มี 'embed_fc.*' → ใช้ Identity")
        load_forgiving(eff, sd, ignore_prefixes=("model.classifier.1.",))
    else:
        print(f"[Warn] ไม่พบ weights EfficientNet ที่ {args.w_eff}")

    # Extract
    print("→ Extracting embeddings (Siamese)"); g_feat_siam = extract_embeddings(siamese, dl_g_siam, device); p_feat_siam = extract_embeddings(siamese, dl_p_siam, device)
    print("→ Extracting embeddings (ViT)");     g_feat_vit  = extract_embeddings(vit, dl_g_vit,  device); p_feat_vit  = extract_embeddings(vit, dl_p_vit,  device)
    print("→ Extracting embeddings (EffB0)");   g_feat_eff  = extract_embeddings(eff, dl_g_eff,  device); p_feat_eff  = extract_embeddings(eff, dl_p_eff,  device)

    if args.print_uniq:
        for n, F in [("Siamese-G", g_feat_siam), ("Siamese-P", p_feat_siam),
                     ("ViT-G", g_feat_vit), ("ViT-P", p_feat_vit),
                     ("Eff-G", g_feat_eff), ("Eff-P", p_feat_eff)]:
            uniq = np.unique(np.round(F, 3), axis=0).shape[0]
            print(f"[dbg/{n}] shape={F.shape}, unique_rows@1e-3={uniq}, std={F.std():.6f}")

    # Verification
    print("→ Verification metrics")
    def run_ver(name, gf, pf):
        y_true, y_score = build_verification_pairs(gf, g_ids, pf, p_ids, neg_per_pos=1, seed=42)
        eer, far, frr, thr = far_frr_eer_from_scores(y_true, y_score)
        tn, fp, fn, tp = confusion_matrix_2x2_from_scores(y_true, y_score, thr)
        with open(os.path.join(args.save_dir, f"{name}_roc.pkl"), "wb") as f: pickle.dump([y_true.tolist(), y_score.tolist()], f)
        with open(os.path.join(args.save_dir, f"{name}_cm_verif.pkl"), "wb") as f: pickle.dump([[tn, fp], [fn, tp]], f)
        acc_at_eer = (tp + tn) / max(1, (tp + tn + fp + fn))
        return {"model": name, "eer": eer, "far_at_eer": far, "frr_at_eer": frr, "thr_eer": thr,
                "acc_at_eer": acc_at_eer, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
    rows = [run_ver("SiameseCNN", g_feat_siam, p_feat_siam),
            run_ver("ViT", g_feat_vit, p_feat_vit),
            run_ver("EfficientNetB0", g_feat_eff, p_feat_eff)]
    df_ver = pd.DataFrame(rows); df_ver.to_csv(os.path.join(args.save_dir, "metrics_verification.csv"), index=False); print(df_ver)

    # Identification
    print("→ Identification (closed-set)")
    def run_ident(name, gf, pf):
        top1, y_pred = topk_identification(gf, g_ids, pf, p_ids, k=1)
        with open(os.path.join(args.save_dir, f"{name}_cm_ident.pkl"), "wb") as f: pickle.dump([p_ids, y_pred], f)
        return {"model": name, "top1_acc": top1}
    rows = [run_ident("SiameseCNN", g_feat_siam, p_feat_siam),
            run_ident("ViT", g_feat_vit, p_feat_vit),
            run_ident("EfficientNetB0", g_feat_eff, p_feat_eff)]
    df_ident = pd.DataFrame(rows); df_ident.to_csv(os.path.join(args.save_dir, "metrics_identification.csv"), index=False); print(df_ident)

if __name__ == "__main__":
    main()
