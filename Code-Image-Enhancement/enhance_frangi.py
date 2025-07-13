import os
import cv2
import numpy as np
from skimage.filters import frangi # Frangi filter สำหรับเน้นเส้นเลือด/โครงสร้างเส้นในภาพ

input_root_dir = 'data/Resize-Img-224'
output_root_dir = 'data/Enhanced-Frangi'
os.makedirs(output_root_dir, exist_ok=True)

def apply_frangi_vesselness_only(gray_img_uint8):
    gray_img_float = gray_img_uint8.astype(np.float32) / 255.0 # แปลงเป็น float 32 ในช่วง 0-1 เพราะ frangi รับแค่ float
    frangi_output_float = frangi(gray_img_float, scale_range=(1, 3), scale_step=1) # Apply Frangi filter scale range 1-3
    frangi_output_normalized = cv2.normalize(frangi_output_float, None, 0, 255, cv2.NORM_MINMAX) # ปรับคืนเป็นช่วง 0-255
    return frangi_output_normalized.astype(np.uint8) # แปลงคืนเป็น uint8

for split_name in ['Train', 'Val', 'Test']:
    input_split_path = os.path.join(input_root_dir, split_name)
    output_split_path = os.path.join(output_root_dir, split_name)

    for subject_id in os.listdir(input_split_path):
        input_id_path = os.path.join(input_split_path, subject_id)
        output_id_path = os.path.join(output_split_path, subject_id)
        os.makedirs(output_id_path, exist_ok=True)

        for filename in os.listdir(input_id_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_image_path = os.path.join(input_id_path, filename)
                output_image_path = os.path.join(output_id_path, filename)

                gray_input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
                if gray_input_image is None:
                    print(f"โหลดภาพไม่ได้ !!! : {input_image_path}")
                    continue

                frangi_only_image = apply_frangi_vesselness_only(gray_input_image)
                cv2.imwrite(output_image_path, frangi_only_image)

print("Frangi-only enhancement เสร็จแล้ว!")
print("เก็บไว้ที่: data/Enhanced-Frangi")
