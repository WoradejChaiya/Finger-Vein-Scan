
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # เพิ่ม path ให้ Python เห็นโฟลเดอร์โปรเจกต์หลัก

import torch  # ใช้ในการสร้าง tensor และจัดการ GPU
from torch.utils.data import DataLoader  # สำหรับโหลดข้อมูลเป็น batch
import pandas as pd  # ใช้โหลดไฟล์ CSV ของ dataset

from datasets.siamese_cnn_dataset import SiameseCnnDataset  # custom dataset สำหรับ Siamese CNN
from models.siamese_cnn_model import SiameseCnnModel  # สถาปัตยกรรม Siamese CNN
from models.siamese_cnn_loss import ContrastiveLoss  # loss function (contrastive loss)
from transforms.transforms_config import transform_pipeline  # preprocessing pipeline

# === CONFIG ===
BATCH_SIZE = 4
CSV_PATH = "data/split_combined.csv"  # เลือก CSV ที่อ้างอิงภาพ Enhanced-Combined
IMAGE_DIR = "data/Enhanced-Combined"  # โฟลเดอร์หลักของภาพที่ preprocess แล้ว (Combined)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ใช้ GPU ถ้ามี มิฉะนั้นใช้ CPU

# === โหลดข้อมูลจาก CSV ===
df = pd.read_csv(CSV_PATH)

# ฟังก์ชันช่วยค้นหา filepath ของภาพ ในกรณีโครงสร้างหลากหลาย
def resolve_image_path(row):
    orig_path = row["filepath"]  # Path ต้นทางจาก CSV
    # กรณี path นั้นมีอยู่จริง (เป็น absolute หรือ relative)
    if os.path.exists(orig_path):
        return orig_path
    # กรณี CSV เก็บ path แบบ relative แต่ลืม prefix 'data/'
    data_rel = os.path.join("data", orig_path)
    if os.path.exists(data_rel):
        return data_rel
    # Flat: image เก็บใน IMAGE_DIR โดยตรง
    fname = os.path.basename(orig_path)
    flat = os.path.join(IMAGE_DIR, fname)
    if os.path.exists(flat):
        return flat
    # Nested: image เก็บใน IMAGE_DIR/<id>/filename
    nested = os.path.join(IMAGE_DIR, str(row["id"]), fname)
    if os.path.exists(nested):
        return nested
    # Recursive search: สแกนหาทั่ว IMAGE_DIR
    for root, dirs, files in os.walk(IMAGE_DIR):
        if fname in files:
            return os.path.join(root, fname)
    # ไม่พบไฟล์ไหนเลย -> แจ้ง error ชัดเจน
    raise FileNotFoundError(f"ไม่พบไฟล์ {fname} ใน '{IMAGE_DIR}' หรือ subdirectories")

# ปรับค่า filepath ใน DataFrame โดยใช้ฟังก์ชันด้านบน
df["filepath"] = df.apply(resolve_image_path, axis=1)

# เตรียม lists สำหรับ Dataset
image_paths = df["filepath"].tolist()
labels = df["id"].values

# === สร้าง Dataset และ DataLoader ===
dataset = SiameseCnnDataset(image_paths=image_paths, labels=labels, transform=transform_pipeline)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === สร้างโมเดลและ loss ===
model = SiameseCnnModel().to(DEVICE)
criterion = ContrastiveLoss()

# === ทดสอบ การ train 1 batch ===
print("== Testing SiameseCNN training step ==")
model.train()
for img1, img2, label in dataloader:
    img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
    out1, out2 = model(img1, img2)
    loss = criterion(out1, out2, label)
    print(f"Batch loss: {loss.item():.4f}")
    break  # แค่ batch แรกพอ
print("✅ SiameseCNN training test complete.")
