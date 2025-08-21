# datasets/vit_dataset.py
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ViTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, labels, transform=None, image_col: str = "filepath"):
        self.df = df.reset_index(drop=True).copy()   # ใช้ตำแหน่งเสมอ
        self.labels = np.asarray(labels, dtype=int)
        self.transform = transform

        if image_col not in self.df.columns:
            for c in ["path", "image_path", "img_path", "file"]:
                if c in self.df.columns:
                    image_col = c; break
        self.image_col = image_col
        assert self.image_col in self.df.columns, f"ไม่พบคอลัมน์พาธรูป: {self.image_col}"
        assert len(self.df) == len(self.labels), "จำนวนแถว df กับ labels ไม่เท่ากัน"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]                     # ใช้ iloc (ตำแหน่ง)
        img_path = str(row[self.image_col])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {img_path}")
        img = Image.open(img_path).convert("L")     # grayscale; transform จะ Grayscale(3) ต่อ
        if self.transform:
            img = self.transform(img)
        label = int(self.labels[idx])
        return img, label
