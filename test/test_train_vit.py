import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # เพิ่ม path ให้ Python มองเห็น project root

import torch  # สำหรับ tensor และ GPU
from torch.utils.data import DataLoader  # สำหรับโหลดข้อมูลเป็น batch
import pandas as pd  # สำหรับอ่าน CSV

from datasets.vit_dataset import ViTDataset  # custom dataset สำหรับ ViT
from models.vit_model import ViTModel  # สถาปัตยกรรม Vision Transformer
from torch.nn import CrossEntropyLoss as ClassificationLoss  # loss function สำหรับ classification
from transforms.vit_transforms import vit_transform_pipeline as vit_transform  # preprocessing pipeline สำหรับ ViT

# === CONFIG ===
BATCH_SIZE = 8  # ปรับตามกำลัง GPU ของคุณ
CSV_PATH = "data/split_combined.csv"  # CSV อ้างอิงภาพ Enhanced-Combined
IMAGE_DIR = "data/Enhanced-Combined"  # โฟลเดอร์หลักภาพที่ preprocess แล้ว
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === โหลด CSV ===
df = pd.read_csv(CSV_PATH)

# ฟังก์ชันช่วยค้นหา filepath ของภาพ (flat, nested, recursive)
def resolve_image_path(row):
    orig = row["filepath"]
    # เช็ค path ตรง ๆ
    if os.path.exists(orig):
        return orig
    # เติม prefix 'data/'
    pref = os.path.join("data", orig)
    if os.path.exists(pref):
        return pref
    # flat: IMAGE_DIR/filename
    fname = os.path.basename(orig)
    flat = os.path.join(IMAGE_DIR, fname)
    if os.path.exists(flat):
        return flat
    # nested: IMAGE_DIR/<id>/filename
    nested = os.path.join(IMAGE_DIR, str(row["id"]), fname)
    if os.path.exists(nested):
        return nested
    # fallback: scan recursive
    for root, _, files in os.walk(IMAGE_DIR):
        if fname in files:
            return os.path.join(root, fname)
    # ถ้าไม่เจอเลย
    raise FileNotFoundError(f"ไม่พบไฟล์ {fname} ใน '{IMAGE_DIR}' หรือ subdirectories")

# ปรับ filepath ใน DataFrame ให้เป็น full path
df["filepath"] = df.apply(resolve_image_path, axis=1)

# สร้าง mapping id -> index
unique_ids = sorted(df["id"].unique())
id_to_idx = {v: i for i, v in enumerate(unique_ids)}
# สร้างคอลัมน์ label index
df["label_idx"] = df["id"].map(id_to_idx)
num_classes = len(unique_ids)

# เตรียม lists สำหรับ Dataset
image_paths = df["filepath"].tolist()
labels = df["label_idx"].values

# === สร้าง Dataset และ DataLoader ===
dataset = ViTDataset(
    image_paths=image_paths,
    labels=labels,
    transform=vit_transform
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === สร้างโมเดลและ loss ===
model = ViTModel(num_classes=num_classes).to(DEVICE)  # ส่ง num_classes ให้โมเดล
criterion = ClassificationLoss()

# === ทดสอบ training 1 batch ===
print("== Testing ViT training step ==")
model.train()
for img, label in dataloader:
    img, label = img.to(DEVICE), label.to(DEVICE)
    output = model(img)  # forward pass
    loss = criterion(output, label)  # คำนวณ loss
    print(f"Batch loss: {loss.item():.4f}")
    break  # แค่ batch แรกก็พอ
print("✅ ViT training test complete.")
