import torch  # ใช้งาน PyTorch สำหรับ tensor และการคำนวณต่างๆ
from torch.utils.data import DataLoader  # ใช้สร้างตัวโหลดข้อมูล (batching/shuffling)
import torch.optim as optim  # เรียกใช้งาน optimizer ของ PyTorch
from torch.optim.lr_scheduler import ReduceLROnPlateau  # เรียก scheduler ปรับ learning rate อัตโนมัติ

from datasets.vit_dataset import ViTDataset  # import คลาส ViTDataset ที่เราสร้างไว้
from models.vit_model import ViTModel  # import โมเดล ViT ที่เราสร้างไว้
from transforms.vit_transforms import vit_transform_pipeline  # import pipeline สำหรับแปลงภาพ

# เตรียมข้อมูล (ใส่ image_paths กับ labels จริงตาม dataset)
image_paths = [...]  # เช่น ['data/Enhanced-Combined/img1.png', ...]  # list ของ path ภาพแต่ละภาพ
labels = [...]       # เช่น [0, 0, 1, 1, ...]  # list ของ label ที่ตรงกับแต่ละภาพ

train_dataset = ViTDataset(image_paths, labels, transform=vit_transform_pipeline)  # สร้าง dataset จาก path/label ที่เตรียมไว้และแปลงภาพตาม pipeline
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # สร้าง DataLoader สำหรับแบ่ง batch และ shuffle ข้อมูล

# กำหนดโมเดล, optimizer, loss, scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # เลือกใช้ GPU ถ้ามี, ไม่งั้นใช้ CPU
num_classes = len(set(labels))  # นับจำนวนคลาสทั้งหมดใน labels
model = ViTModel(num_classes=num_classes).to(device)  # สร้างโมเดล ViT และส่งไปยังอุปกรณ์ที่เลือก

criterion = nn.CrossEntropyLoss()  # สร้าง loss function แบบ cross entropy สำหรับงาน classification
optimizer = optim.AdamW(model.parameters(), lr=1e-4)  # สร้าง optimizer แบบ AdamW สำหรับปรับ weight ของโมเดล
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)  # ถ้า loss ไม่ลดลงต่อเนื่อง 3 epoch จะลด learning rate ให้อัตโนมัติ

# Training loop
num_epochs = 10  # กำหนดจำนวนรอบ epoch ที่จะ train
for epoch in range(num_epochs):  # วน loop ตามจำนวน epoch
    model.train()  # เซ็ตโมเดลให้อยู่ในโหมด train
    total_loss = 0  # รีเซ็ตตัวแปรสำหรับสะสม loss แต่ละ epoch
    for images, targets in train_loader:  # วนลูปผ่าน batch ใน train_loader
        images, targets = images.to(device), targets.to(device)  # ส่งข้อมูลและ label ไปที่ device (GPU/CPU)
        
        optimizer.zero_grad()  # เคลียร์ gradient เก่าก่อนเริ่มรอบใหม่
        outputs = model(images)  # ทำ forward pass ผ่านโมเดล
        loss = criterion(outputs, targets)  # คำนวณ loss ระหว่าง output กับ label จริง
        loss.backward()  # คำนวณ gradient (backpropagation)
        optimizer.step()  # ปรับ weights ของโมเดลตาม gradient ที่ได้

        total_loss += loss.item()  # สะสมค่าความสูญเสีย (loss) ของ batch นี้

    avg_loss = total_loss / len(train_loader)  # คำนวณ loss เฉลี่ยทั้ง epoch
    scheduler.step(avg_loss)  # ปรับ learning rate ตาม loss เฉลี่ย

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")  # แสดงผลลัพธ์ loss ของแต่ละ epoch
