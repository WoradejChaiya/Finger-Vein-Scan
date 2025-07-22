import torch  # เรียกใช้ไลบรารี PyTorch สำหรับจัดการ tensor และ deep learning
from torch.utils.data import Dataset  # เรียกใช้คลาส Dataset จาก PyTorch เพื่อสร้าง dataset เอง
from PIL import Image  # เรียกใช้ PIL สำหรับจัดการรูปภาพ

class ViTDataset(Dataset):  # สร้างคลาสชื่อ ViTDataset โดยสืบทอดจาก Dataset ของ PyTorch
    def __init__(self, image_paths, labels, transform=None):  # ฟังก์ชันกำหนดค่าเริ่มต้นเมื่อสร้างออบเจกต์จากคลาสนี้
        self.image_paths = image_paths  # เก็บ list ของ paths ของไฟล์ภาพไว้ในตัวแปร image_paths
        self.labels = labels  # เก็บ list ของ labels ของแต่ละภาพ
        self.transform = transform  # เก็บ transform pipeline (เช่น resizing, normalization) ที่จะใช้กับภาพ

    def __len__(self):  # ฟังก์ชันที่คืนจำนวนข้อมูลใน dataset
        return len(self.image_paths)  # คืนค่าจำนวนภาพทั้งหมดใน dataset

    def __getitem__(self, idx):  # ฟังก์ชันสำหรับเรียกข้อมูลตัวอย่างจาก dataset ตาม index ที่กำหนด
        img_path = self.image_paths[idx]  # ดึง path ของภาพที่ index ที่กำหนด
        label = self.labels[idx]  # ดึง label ของภาพที่ index ที่กำหนด

        image = Image.open(img_path).convert('L')  # เปิดไฟล์ภาพและแปลงเป็น grayscale ('L')

        if self.transform:  # ตรวจสอบว่ามี transform pipeline กำหนดไว้หรือไม่
            image = self.transform(image)  # ถ้ามี ให้แปลงภาพตาม pipeline ที่กำหนดไว้

        return image, torch.tensor(label, dtype=torch.long)  # คืนค่าเป็นภาพที่ถูกแปลงแล้ว กับ label ของภาพในรูปแบบ tensor ประเภท long (int64)
