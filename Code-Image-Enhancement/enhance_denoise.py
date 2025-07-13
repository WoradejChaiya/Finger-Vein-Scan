import os
import cv2
import numpy as np

input_root = 'data/Resize-Img-224'
output_root = 'data/Enhanced-Denoise'
os.makedirs(output_root, exist_ok=True)

def apply_denoise(img, ksize=3):
    # ใช้ median blur ksize3*3 เพื่อลด noise
    return cv2.medianBlur(img, ksize) 

for split in ['Train', 'Val', 'Test']:
    input_split_path = os.path.join(input_root, split)
    output_split_path = os.path.join(output_root, split)

    for id_folder in os.listdir(input_split_path):
        id_input_path = os.path.join(input_split_path, id_folder)
        id_output_path = os.path.join(output_split_path, id_folder)
        os.makedirs(id_output_path, exist_ok=True)

        for filename in os.listdir(id_input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_img_path = os.path.join(id_input_path, filename)
                output_img_path = os.path.join(id_output_path, filename)

                img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"โหลดภาพไม่ได้ !!! : {input_img_path}")
                    continue

                enhanced_img = apply_denoise(img)
                cv2.imwrite(output_img_path, enhanced_img)

print("Denoise enhancement เสร็จแล้ว!")
print("เก็บไว้ที่: data/Enhanced-Denoise")
