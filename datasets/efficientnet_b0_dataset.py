import torch  # นำเข้าไลบรารีหลักของ PyTorch สำหรับการสร้าง tensor และฟังก์ชันต่างๆ
from torch.utils.data import Dataset  # นำเข้า Dataset class เพื่อสืบทอดและสร้างชุดข้อมูลสำหรับ DataLoader
from PIL import Image  # นำเข้า Image จาก PIL สำหรับเปิดไฟล์ภาพและแปลงรูปแบบ

class EfficientNetB0Dataset(Dataset):  # สร้างคลาสชุดข้อมูลสำหรับ EfficientNetB0 โดยสืบทอดจาก torch.utils.data.Dataset
    def __init__(self, image_paths, labels, transform=None):  # กำหนดเมธอดสร้างวัตถุ รับพารามิเตอร์เป็น list ของ path รูป, labels, และ optional transform
        self.image_paths = image_paths  # เก็บ list ของเส้นทางภาพทั้งหมด
        self.labels = labels  # เก็บ list ของป้ายกำกับ (labels) ที่สอดคล้องกับภาพแต่ละไฟล์
        self.transform = transform  # เก็บชุดแปลงภาพ (เช่น normalization, augmentation) ถ้ามี

    def __len__(self):  # เมธอดที่ DataLoader เรียกใช้ เพื่อทราบจำนวนตัวอย่างในชุดข้อมูล
        return len(self.image_paths)  # คืนจำนวนภาพทั้งหมดในชุดข้อมูล

    def __getitem__(self, idx):  # เมธอดที่ DataLoader เรียกใช้ เพื่อดึงตัวอย่างที่ตำแหน่ง idx
        img_path = self.image_paths[idx]  # ดึง path ของภาพตาม index
        label = self.labels[idx]  # ดึงป้ายกำกับที่สอดคล้องกับภาพนั้น

        image = Image.open(img_path).convert('L')  # เปิดไฟล์ภาพและแปลงเป็นโหมด grayscale (ช่องเดี่ยว) ด้วย convert('L')
        if self.transform:  # ถ้ามีการกำหนด transform มา
            image = self.transform(image)  # นำภาพเข้า pipeline ของ transform

        return image, torch.tensor(label, dtype=torch.long)  # คืนภาพ (หลัง transform) และ label ในรูป tensor ของ PyTorch
