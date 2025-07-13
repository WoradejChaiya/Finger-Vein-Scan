import os
import cv2
import numpy as np

# โฟลเดอร์ต้นทางและปลายทาง
input_root = 'data/Raw-Img'  # ภาพ Input
output_root = 'data/Resize-Img-224'  # ปลายทาง Output
target_size = (224, 224)  # ขนาดที่จะปรับ

os.makedirs(output_root, exist_ok=True)

def resize_with_padding(img, target_size=(224, 224)):
    h, w = img.shape[:2]  # ดึง กว้างยาวจากภาพ (ไม่เอาcolor channel)
    target_w, target_h = target_size # ตั้งชื่อแยกกัน

    scale = min(target_w / w, target_h / h)  # min เพื่อให้ ภาพขนาดใหม่ไม่เกิน   และ aspect ratio ภาพ
    new_w, new_h = int(w * scale), int(h * scale)  # ขนาดภาพใหม่หลัง resize จาก 600*300 เป็น 224*112

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # ย่อภาพโดยใช้ cv2.INTER_AREA(interpolate(หาค่าพิกเซลหลังย่อ))

    result = np.zeros((target_h, target_w), dtype=np.uint8)  # สร้างภาพดำ(ใช้เป็น BG) uint8 เหมาะกับ grayscale
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2  # หาตำแหน่งกึ่งกลาง
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img  ## วางภาพเป็นกึ่งกลางที่ย่อบนภาพสีดำ

    return result # save

# loop ผ่าน Train, Val, Test
for split in ['Train', 'Val', 'Test']:
    input_split_path = os.path.join(input_root, split)  #  สร้าง path เช่น data/Raw-Img/Train
    output_split_path = os.path.join(output_root, split)  #  สร้าง path เช่น data/Resize-Img-225/Train

    for id_folder in os.listdir(input_split_path):
        id_input_path = os.path.join(input_split_path, id_folder)  # สร้าง path ID
        id_output_path = os.path.join(output_split_path, id_folder)  # สร้าง path ID

        os.makedirs(id_output_path, exist_ok=True)  # สร้าง folder เก็บภาพหลัง resize

        for filename in os.listdir(id_input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_img_path = os.path.join(id_input_path, filename)
                output_img_path = os.path.join(id_output_path, filename)

                img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️ อ่านภาพไม่ได้: {input_img_path}")  # เช็คว่าภาพโหลดได้จริง  ถ้าไม่ให้ข้ามทันที เพื่อไม่ให้ crash
                    continue 

                processed_img = resize_with_padding(img, target_size)
                cv2.imwrite(output_img_path, processed_img)

print("✅ Resize + Padding เสร็จแล้ว! 🔄")

