import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # เพิ่ม path ให้ Python เห็นโฟลเดอร์โปรเจกต์หลัก

import torch  # สำหรับ tensor และ GPU
from torch.utils.data import DataLoader  # สำหรับโหลดข้อมูลเป็น batch
import pandas as pd  # สำหรับอ่าน CSV และจัดการ DataFrame

from datasets.efficientnet_b0_dataset import EfficientNetB0Dataset  # custom dataset สำหรับ EfficientNet-B0
from models.efficientnet_b0_model import EfficientNetB0Model  # สถาปัตยกรรม EfficientNet-B0
from torch.nn import CrossEntropyLoss as ClassificationLoss  # ใช้ CrossEntropyLoss สำหรับ classification
from transforms.efficientnet_b0_transforms import transform_pipeline as efficientnet_transform  # preprocessing pipeline

# === CONFIG ===
BATCH_SIZE = 16  # ปรับตามกำลัง GPU ของคุณ
CSV_PATH = "data/split_combined.csv"  # เลือกไฟล์ CSV ที่อ้างอิงภาพ Enhanced-Combined
IMAGE_DIR = "data/Enhanced-Combined"  # โฟลเดอร์หลักของภาพหลัง combined
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === โหลด CSV และเตรียม paths ===
df = pd.read_csv(CSV_PATH)

# ฟังก์ชันช่วยค้นหา filepath ของภาพ (flat, nested, recursive)
def resolve_image_path(row):
    orig = row["filepath"]
    if os.path.exists(orig):
        return orig
    prefixed = os.path.join("data", orig)
    if os.path.exists(prefixed):
        return prefixed
    fname = os.path.basename(orig)
    flat = os.path.join(IMAGE_DIR, fname)
    if os.path.exists(flat):
        return flat
    nested = os.path.join(IMAGE_DIR, str(row["id"]), fname)
    if os.path.exists(nested):
        return nested
    for root, _, files in os.walk(IMAGE_DIR):
        if fname in files:
            return os.path.join(root, fname)
    raise FileNotFoundError(f"ไม่พบไฟล์ {fname} ใน '{IMAGE_DIR}' หรือ subdirectories")

# ปรับ filepath ใน DataFrame และสร้าง label indices
# แปะ path ใหม่
df["filepath"] = df.apply(resolve_image_path, axis=1)
# สร้าง mapping id -> class index (0..num_classes-1)
unique_ids = sorted(df["id"].unique())
id_to_idx = {id_val: idx for idx, id_val in enumerate(unique_ids)}
# สร้างคอลัมน์ label index ใหม่
df["label_idx"] = df["id"].map(id_to_idx)

# เตรียม lists สำหรับ Dataset
image_paths = df["filepath"].tolist()  # list ของ full paths
labels = df["label_idx"].values        # list ของ indices
num_classes = len(unique_ids)            # จำนวน classes

# === สร้าง Dataset และ DataLoader ===
dataset = EfficientNetB0Dataset(
    image_paths=image_paths,
    labels=labels,
    transform=efficientnet_transform
)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# === สร้างโมเดลและ loss ===
model = EfficientNetB0Model(num_classes=num_classes).to(DEVICE)  # ส่ง num_classes ให้โมเดล
criterion = ClassificationLoss()

# === ทดสอบ train 1 batch ===
print("== Testing EfficientNet-B0 training step ==")
model.train()
for img, label in dataloader:
    img, label = img.to(DEVICE), label.to(DEVICE)
    output = model(img)                    # forward pass
    loss = criterion(output, label)       # คำนวณ loss
    print(f"Batch loss: {loss.item():.4f}")
    break  # ทดสอบแค่ 1 batch
print("✅ EfficientNet-B0 training test complete.")