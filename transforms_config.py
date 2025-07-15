import torchvision.transforms as T

# Transform สำหรับชุด Train (ใช้ data augmentation)
train_transforms = T.Compose([
    T.RandomHorizontalFlip(p=0.5),   # พลิกภาพซ้าย-ขวาแบบสุ่ม
    T.RandomRotation(degrees=5),     # หมุนภาพเล็กน้อย
    T.ToTensor(),                    # แปลงจาก PIL เป็น Tensor [C, H, W]
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize ให้ค่าอยู่ในช่วง [-1, 1]
])

# Transform สำหรับ Val/Test (ไม่ใช้การสุ่ม)
test_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])
