import torchvision.transforms as T  # เรียกใช้โมดูล transforms จาก torchvision สำหรับแปลงภาพ

vit_transform_pipeline = T.Compose([  # สร้าง pipeline สำหรับแปลงภาพ โดยรวมหลายๆ transform ต่อกัน
    T.Resize((224, 224)),             # ปรับขนาดภาพเป็น 224x224 พิกเซล (ขนาดที่ ViT ใช้)
    T.ToTensor(),                     # แปลงภาพจาก PIL เป็น tensor (ค่าพิกเซล 0-1)
    T.Normalize(mean=[0.5], std=[0.5])])  # ปรับ normalize ค่า pixel ให้มี mean=0.5, std=0.5 (กับภาพขาวดำ)

