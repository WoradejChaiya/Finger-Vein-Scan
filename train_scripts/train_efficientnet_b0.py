import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch  # เรียกใช้ไลบรารี PyTorch สำหรับ tensor และ deep learning
from torch.utils.data import DataLoader  # import DataLoader สำหรับโหลด batch ของข้อมูล
import torch.optim as optim  # import optimizer (เช่น AdamW)
from torch.optim.lr_scheduler import ReduceLROnPlateau  # import scheduler สำหรับปรับ learning rate อัตโนมัติ

from datasets.efficientnet_b0_dataset import EfficientNetB0Dataset  # import dataset class ที่ custom สำหรับ EfficientNetB0
from models.efficientnet_b0_model import EfficientNetB0Model  # import โมเดล EfficientNetB0 ที่ custom
from models.efficientnet_b0_loss import efficientnet_b0_loss_fn  # import loss function ที่ใช้กับ EfficientNetB0
from transforms.efficientnet_b0_transforms import efficientnet_b0_transforms  # import transform pipeline สำหรับ EfficientNetB0

# ====== เตรียมข้อมูล ======
image_paths = [...]  # เช่น ['data/Enhanced-Combined/img1.png', ...]  # กำหนด list path ของไฟล์ภาพ
labels = [...]       # เช่น [0, 0, 1, 1, ...]  # กำหนด list label ของแต่ละภาพ

dataset = EfficientNetB0Dataset(image_paths, labels, transform=efficientnet_b0_transforms)  # สร้าง dataset พร้อมแปลงภาพด้วย pipeline ที่กำหนด
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)  # โหลด dataset แบบแบ่ง batch ขนาด 32, shuffle ข้อมูล และใช้ multi-thread

# ====== เตรียมโมเดล ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # กำหนดว่าใช้ GPU ถ้ามี ไม่งั้นใช้ CPU
num_classes = len(set(labels))  # นับจำนวนคลาสที่มีใน label
model = EfficientNetB0Model(num_classes=num_classes).to(device)  # สร้างโมเดล EfficientNetB0 และส่งไปที่อุปกรณ์ที่กำหนด

# ====== Loss, Optimizer, Scheduler ======
criterion = efficientnet_b0_loss_fn  # ใช้ loss function ที่ import มา
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # กำหนด optimizer แบบ AdamW และ learning rate = 1e-4
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)  # ถ้า loss ไม่ลดลง 3 รอบ จะลด learning rate

# ====== Training Loop ======
num_epochs = 10  # กำหนดจำนวนรอบ epoch ที่จะ train
for epoch in range(num_epochs):  # วนลูปตามจำนวน epoch
    model.train()  # ตั้งโมเดลให้อยู่ในโหมด train
    running_loss = 0.0  # ตัวแปรสะสมค่า loss ในแต่ละ epoch

    for images, targets in loader:  # วนลูปผ่าน batch ของข้อมูล
        images, targets = images.to(device), targets.to(device)  # ส่งข้อมูลภาพและ label ไปที่อุปกรณ์ (GPU/CPU)

        optimizer.zero_grad()  # ลบ gradient เก่า
        outputs = model(images)  # ส่งภาพเข้าโมเดลเพื่อให้โมเดลทำนายผล
        loss = criterion(outputs, targets)  # คำนวณ loss ระหว่าง output กับ target
        loss.backward()  # คำนวณ gradient (backpropagation)
        optimizer.step()  # อัปเดต weight ของโมเดล

        running_loss += loss.item()  # สะสมค่า loss ของแต่ละ batch

    avg_loss = running_loss / len(loader)  # คำนวณ loss เฉลี่ยทั้ง epoch
    scheduler.step(avg_loss)  # ส่งค่า avg_loss ให้ scheduler เพื่อตรวจสอบว่าต้องลด learning rate หรือไม่
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")  # แสดงผลลัพธ์ loss ของแต่ละ epoch
