import sys, os
sys.path.append(os.path.abspath("."))

import torch
import pandas as pd
from torch.utils.data import DataLoader

from datasets.siamese_cnn_dataset import SiameseCnnDataset
from transforms.transforms_config import transform_pipeline as siamese_transform

from datasets.vit_dataset import ViTDataset
from transforms.vit_transforms import vit_transform_pipeline as vit_transform

from datasets.efficientnet_b0_dataset import EfficientNetB0Dataset
from transforms.efficientnet_b0_transforms import transform_pipeline as eff_transform

def main():
    CSV_PATH = "data/split_combined.csv"
    df = pd.read_csv(CSV_PATH)
    df = df.sample(128, random_state=42).reset_index(drop=True)
    unique_ids = sorted(df["id"].unique())
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    df["label_idx"] = df["id"].map(id_to_idx)

    print("=== SiameseCNN DataLoader ===")
    siam_ds = SiameseCnnDataset(df["filepath"], df["label_idx"], siamese_transform)
    siam_loader = DataLoader(siam_ds, batch_size=16, num_workers=2)
    for img1, img2, lbl in siam_loader:
        print("img1 shape:", img1.shape)
        print("img2 shape:", img2.shape)
        print("lbl shape:", lbl.shape)
        print("lbl:", lbl[:10])
        break

    print("=== ViT DataLoader ===")
    vit_ds = ViTDataset(df["filepath"], df["label_idx"], vit_transform)
    vit_loader = DataLoader(vit_ds, batch_size=16, num_workers=2)
    for img, lbl in vit_loader:
        print("img shape:", img.shape)
        print("lbl shape:", lbl.shape)
        print("lbl:", lbl[:10])
        break

    print("=== EfficientNetB0 DataLoader ===")
    eff_ds = EfficientNetB0Dataset(df["filepath"], df["label_idx"], eff_transform)
    eff_loader = DataLoader(eff_ds, batch_size=16, num_workers=2)
    for img, lbl in eff_loader:
        print("img shape:", img.shape)
        print("lbl shape:", lbl.shape)
        print("lbl:", lbl[:10])
        break

if __name__ == "__main__":
    main()
