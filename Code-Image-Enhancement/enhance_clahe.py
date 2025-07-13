import os
import cv2
import numpy as np

input_root = 'data/Resize-Img-224' 
output_root = 'data/Enhanced-CLAHE'

os.makedirs(output_root, exist_ok=True) # สร้างโฟลเดอร์ output_root     exist_ok=True ป้องกัน error ถ้ามีอยู่แล้ว

# สร้าง CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  
# clipLimit=2.0 จำกัด contrast เพื่อไม่ให้เกิด noise มากเกินไป
# tileGridSize 0=   (8, 8)  แบ่งภาพเป็น grid 8x8 แล้วทำ histogram equalization แยกแต่ละ tile

def apply_clahe(img):
    return clahe.apply(img)

# วนรอบ Train / Val / Test
for split in ['Train', 'Val', 'Test']:
    input_split_path = os.path.join(input_root, split)
    output_split_path = os.path.join(output_root, split) # สร้าง path สำหรับโฟลเดอร์ย่อย imput & output

    for id_folder in os.listdir(input_split_path): # อ่านรายชื่อโฟลเดอร์ (ID) ใน Train Val Test
        id_input_path = os.path.join(input_split_path, id_folder) # เช่น data/Resize-Img-224/Train/ID001
        id_output_path = os.path.join(output_split_path, id_folder) # เช่น data/Enhanced-CLAHE/Train/ID001
        os.makedirs(id_output_path, exist_ok=True) # สร้างโฟลเดอร์ปลายทาง

        for filename in os.listdir(id_input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): # (รองรับหลายสกุล)
                input_img_path = os.path.join(id_input_path, filename)
                output_img_path = os.path.join(id_output_path, filename) # สร้าง path ของแต่ละภาพ

                img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE) 
                if img is None: 
                    print(f"โหลดภาพไม่ได้ !!! : {input_img_path}")
                    continue  # ถ้ามีปัญหาหรือโหลดภาพไม่ได้ ให้ขึ้นข้อความเตือนและข้ามไปภาพต่อไป

                enhanced_img = apply_clahe(img)
                cv2.imwrite(output_img_path, enhanced_img) # บันทึกภาพที่ผ่าน CLAHE ไปยัง path output ใช้ชื่อไฟล์เดิม แต่เก็บในโฟลเดอร์ใหม่

print("CLAHE enhancement เสร็จแล้ว!")
print("เก็บไว้ที่: data/Enhanced-CLAHE")
