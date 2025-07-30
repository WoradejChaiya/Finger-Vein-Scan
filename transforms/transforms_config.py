# transforms/transforms_config.py

from torchvision import transforms  # ใช้สำหรับจัดการ preprocessing ภาพ (grayscale)

# สร้าง pipeline สำหรับ Siamese CNN (ใช้กับ grayscale image ขนาด 224x224)
transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # บังคับให้เป็นภาพขาวดำ (1 channel)
    transforms.Resize((224, 224)),  # ย่อขนาดให้เท่ากันทุกภาพ
    transforms.ToTensor(),  # แปลงภาพให้เป็น tensor (ค่า 0.0–1.0)
    transforms.Normalize(mean=[0.5], std=[0.5])  # ปรับค่า pixel ให้อยู่ในช่วง [-1, 1]
])
