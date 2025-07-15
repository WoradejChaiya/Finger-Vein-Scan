import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VeinDataset(Dataset):
    def __init__(self, root_dir, transform=None): # 	รับ path หลัก, เตรียม list image + label
        """
        root_dir: path ไปยัง Train/Val/Test folder เช่น data/Enhanced-FrangiCombined/Train
        transform: torchvision.transforms สำหรับแปลงภาพ เช่น ToTensor(), Normalize
        """
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []   # list path ของภาพทั้งหมด
        self.labels = []        # list label ที่สอดคล้องกับแต่ละภาพ
        self.class_to_idx = {}  # map ชื่อโฟลเดอร์ (เช่น 10801) → label เป็นตัวเลข

        self._scan_dataset()    # เรียก method ที่ใช้เก็บ path และ label

    def _scan_dataset(self):
        # อ่านชื่อ folder ทั้งหมดใน root_dir (สมมุติคือ ['10801', '10802', ...])
        subject_folders = sorted(os.listdir(self.root_dir))
        
        # สร้าง map: '10801' → 0, '10802' → 1, ...
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(subject_folders)}

        for class_name in subject_folders:
            class_folder_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_folder_path):
                continue  # ข้ามถ้าไม่ใช่โฟลเดอร์

            # วนไฟล์ทั้งหมดใน folder นั้น
            for file_name in os.listdir(class_folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    full_img_path = os.path.join(class_folder_path, file_name)
                    self.image_paths.append(full_img_path)
                    self.labels.append(self.class_to_idx[class_name])  # เพิ่ม label ด้วย

    def __len__(self):
        # จำนวนข้อมูลทั้งหมด
        return len(self.image_paths)

    def __getitem__(self, idx): 
        # ดึง path และ label ตาม index
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # โหลดภาพแบบ grayscale ด้วย PIL
        image = Image.open(image_path).convert('L')  # 'L' = grayscale mode

        # ถ้ามี transform (ToTensor, Normalize ฯลฯ) → apply
        if self.transform:
            image = self.transform(image)

        return image, label
