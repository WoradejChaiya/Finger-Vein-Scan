import torch
from torch.utils.data import Dataset # ใช้สำหรับสร้าง Dataset แบบกำหนดเอง
from PIL import Image
import random

class SiameseCnnDataset(Dataset): # สร้างคลาส SiameseCnnDataset ที่สืบทอดจาก PyTorch Dataset
    def __init__(self, image_paths, labels, transform=None): 
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths) # ระบุจำนวนข้อมูลทั้งหมด

    def __getitem__(self, idx): # ดึงข้อมูล
        img1_path = self.image_paths[idx] # ดึง path ของภาพที่ 1 จาก index
        label1 = self.labels[idx] # ดึง label ของภาพที่ 1

        same_class = random.choice([True, False]) # สุ่มว่าจะเลือกภาพที่ 2 จาก class เดียวกันหรือไม่
        if same_class:
            # สุ่ม index ของภาพอื่นที่มี label เดียวกัน (IDเดียวกัน)
            idx2 = random.choice([i for i, lbl in enumerate(self.labels) if lbl == label1]) # เหมือนกัน (label=1)
            label = 1
        else:
            # สุ่ม index ของภาพที่มี label ต่างกัน (IDเดียวกัน)
            idx2 = random.choice([i for i, lbl in enumerate(self.labels) if lbl != label1]) # ต่างกัน (label=0)
            label = 0

        img2_path = self.image_paths[idx2] # ดึง path ของภาพที่ 2

        img1 = Image.open(img1_path).convert('L') # เปิดภาพที่ 1 แล้วแปลงเป็น แปลงเป็น grayscale
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1) # ถ้ามี transform ให้แปลงภาพที่ 1
            img2 = self.transform(img2) # แปลงภาพที่ 2 ด้วย

        return img1, img2, torch.tensor(label, dtype=torch.float32)
        # ส่งคืน: ภาพ 2 ภาพ + label ที่บอกว่าเหมือนกัน (1) หรือไม่ (0)