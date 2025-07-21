import torch                                                       # import PyTorch สำหรับ tensor และ GPU support
from torch.utils.data import DataLoader                            # import DataLoader สำหรับโหลดข้อมูลแบบเป็น batch
import torch.optim as optim                                        # import optimizer เช่น Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau             # scheduler ปรับ learning rate อัตโนมัติเมื่อ loss หยุดลด

from datasets.siamese_cnn_dataset import SiameseCnnDataset         # import Siamese Dataset ที่สร้างขึ้นเอง
from models.siamese_cnn_model import SiameseCnnModel               # import Siamese CNN โมเดลที่ใช้ ResNet
from models.siamese_cnn_loss import siamese_cnn_contrastive_loss   # import ฟังก์ชัน loss แบบ contrastive
from transforms.transforms_config import transform_pipeline        # import transform pipeline ที่เตรียมไว้ล่วงหน้า (เช่น ToTensor, Normalize

# เตรียมข้อมูล (ใส่ list image_paths กับ labels จริงตาม dataset)
image_paths = [...]  # ตัวอย่างเช่น ['data/Enhanced-Combined/img1.png', ...]
labels = [...]       # เช่น [0, 0, 1, 1, ...]

train_dataset = SiameseCnnDataset(image_paths, labels, transform=transform_pipeline)  # สร้าง dataset จาก path และ label พร้อมแปลงภาพ
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)                 # โหลด dataset เป็น batch ขนาด 32 และสลับลำดับทุก epoch

# เตรียมโมเดลและ optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ใช้ GPU ถ้ามี, ไม่งั้น fallback เป็น CPU
model = SiameseCnnModel().to(device)                                   # สร้างโมเดล Siamese และย้ายไปยัง device ที่เลือก
optimizer = optim.Adam(model.parameters(), lr=1e-4)                    # ใช้ Adam optimizer ปรับพารามิเตอร์ของโมเดล
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)  # ถ้า loss ไม่ลดภายใน 3 epoch → ลด learning rate


# Training loop เบื้องต้น
num_epochs = 10                                      # กำหนดจำนวนรอบ epoch ที่จะ train
for epoch in range(num_epochs):                      # วนรอบแต่ละ epoch
    model.train()                                    # เปิดโหมด train สำหรับโมเดล (เปิด dropout/batchnorm)
    total_loss = 0                                   # ค่า loss สะสมราย epoch
        for img1, img2, label in train_loader:           # วนลูปผ่านแต่ละ batch ที่โหลดจาก DataLoader
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)  # ย้ายข้อมูลไปยัง device (GPU หรือ CPU)

        optimizer.zero_grad()                        # เคลียร์ gradient เดิมก่อน backprop
        emb1, emb2 = model(img1, img2)               # forward ภาพคู่ผ่านโมเดล Siamese → ได้ embeddings
        loss = siamese_cnn_contrastive_loss(emb1, emb2, label)  # คำนวณ contrastive loss ระหว่าง embeddings
        loss.backward()                              # คำนวณ gradient ย้อนกลับ
        optimizer.step()                             # ปรับพารามิเตอร์ของโมเดลตาม gradient
        total_loss += loss.item()                    # สะสมค่า loss ของ batch นี้

    avg_loss = total_loss / len(train_loader)        # คำนวณ loss เฉลี่ยต่อ 1 epoch
    scheduler.step(avg_loss)                         # แจ้ง scheduler ให้พิจารณาปรับ learning rate ตาม avg_loss

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")  # แสดงผลการ train ต่อ epoch
