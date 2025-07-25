import torchvision.transforms as T  # เรียกใช้โมดูล transforms จาก torchvision สำหรับแปลงและเตรียมภาพ

efficientnet_b0_transforms = T.Compose([      # สร้าง pipeline สำหรับแปลงภาพ โดยใช้หลาย transforms ต่อเนื่องกัน
    T.Resize((224, 224)),                     # ปรับขนาดภาพให้เป็น 224x224 พิกเซล (ขนาดที่ EfficientNet B0 ใช้)
    T.ToTensor(),                             # แปลงภาพจาก PIL เป็น tensor (ค่าพิกเซลอยู่ระหว่าง 0-1)
    T.Normalize(mean=[0.5], std=[0.5])        # ปรับ normalize ค่าพิกเซล (mean=0.5, std=0.5) สำหรับภาพขาวดำ
])
