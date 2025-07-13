import os
import cv2
import numpy as np
from skimage.filters import frangi
from skimage import img_as_ubyte

# Path
input_root_dir = 'data/Resize-Img-224'
output_root_dir = 'data/Enhanced-Combined'
os.makedirs(output_root_dir, exist_ok=True)

# ---------- STEP 1: CLAHE ----------
def apply_clahe_local_contrast(gray_img_uint8):
    clahe_operator = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe_operator.apply(gray_img_uint8)

# ---------- STEP 2: Frangi Filter ----------
def apply_frangi_vesselness_filter(gray_img_uint8):
    # Convert to float in [0,1]
    gray_img_float = gray_img_uint8.astype(np.float32) / 255.0
    # Apply Frangi filter
    frangi_response_float = frangi(gray_img_float, scale_range=(1, 3), scale_step=1)
    # Normalize to [0, 255] and convert to uint8
    frangi_response_normalized = cv2.normalize(frangi_response_float, None, 0, 255, cv2.NORM_MINMAX)
    return frangi_response_normalized.astype(np.uint8)

# ---------- STEP 3: Median Denoising ----------
def apply_median_denoising(gray_img_uint8, kernel_size=3):
    return cv2.medianBlur(gray_img_uint8, kernel_size)

# ---------- Main Loop ----------
for dataset_split_name in ['Train', 'Val', 'Test']:
    input_split_path = os.path.join(input_root_dir, dataset_split_name)
    output_split_path = os.path.join(output_root_dir, dataset_split_name)

    for subject_id_folder in os.listdir(input_split_path):
        input_id_path = os.path.join(input_split_path, subject_id_folder)
        output_id_path = os.path.join(output_split_path, subject_id_folder)
        os.makedirs(output_id_path, exist_ok=True)

        for image_filename in os.listdir(input_id_path):
            if image_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                input_image_path = os.path.join(input_id_path, image_filename)
                output_image_path = os.path.join(output_id_path, image_filename)

                input_image_gray = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
                if input_image_gray is None:
                    print(f"โหลดภาพไม่ได้ !!! : {input_image_path}")
                    continue

                # ----- Enhancement Pipeline -----
                image_after_clahe = apply_clahe_local_contrast(input_image_gray)
                image_after_frangi = apply_frangi_vesselness_filter(image_after_clahe)
                image_after_denoise = apply_median_denoising(image_after_frangi)

                cv2.imwrite(output_image_path, image_after_denoise)

print("Combined enhancement (CLAHE → Frangi → Denoise) เสร็จแล้ว!")
print("เก็บไว้ที่: data/Enhanced-Combined")
