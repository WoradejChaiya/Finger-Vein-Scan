import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch  # ใช้งาน PyTorch สำหรับ tensor และการคำนวณต่างๆ
import torch.nn as nn  # เพิ่ม nn สำหรับ CrossEntropyLoss
from torch.utils.data import DataLoader  # ใช้สร้างตัวโหลดข้อมูล (batching/shuffling)
import torch.optim as optim  # เรียกใช้งาน optimizer ของ PyTorch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # เรียก scheduler ปรับ learning rate อัตโนมัติ
from tqdm import tqdm  # สำหรับ progress bar

import pandas as pd  # สำหรับอ่าน csv

from datasets.vit_dataset import ViTDataset  # import คลาส ViTDataset ที่เราสร้างไว้
from models.vit_model import ViTModel  # import โมเดล ViT ที่เราสร้างไว้
from transforms.vit_transforms import vit_transform_pipeline  # import pipeline สำหรับแปลงภาพ

# ==== เตรียมข้อมูลจาก .csv ====
df = pd.read_csv('data/split_combined.csv')
train_df = df[df['split'] == 'train']
image_paths = train_df['filepath'].tolist()
id_list = train_df['id'].tolist()

# สร้าง mapping id -> index
unique_ids = sorted(list(set(id_list)))
id2idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
labels = [id2idx[x] for x in id_list]  # map label ทั้งหมดเป็น index (0,1,2,...)

num_classes = len(unique_ids)

# ==== สร้าง Dataset & DataLoader ====
train_dataset = ViTDataset(image_paths, labels, transform=vit_transform_pipeline)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==== กำหนดโมเดล, optimizer, loss, scheduler ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(set(labels))  # จำนวนคลาส
model = ViTModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)  # ตัด verbose=True ถ้า torch <1.6

# ==== Training loop  ====
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    # เพิ่ม tqdm ให้เห็น progress bar ของ batch ในแต่ละ epoch
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
