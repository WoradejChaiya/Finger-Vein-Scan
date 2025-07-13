import os
import cv2
import numpy as np

input_root = 'data/Resize-Img-224'
output_root = 'data/Enhanced-Gabor'
os.makedirs(output_root, exist_ok=True)

def build_gabor_kernels(ksize=21, sigma=5.0, lambd=10.0, gamma=0.5, psi=0):
    # kernel 21x21   sigma ความเบลอ(standard deviation ของ Gaussian)   lambd ความยาวคลื่นของ sinusoidal   gamma อัตราส่วน aspect ratio  

    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4] # มุมที่ใช้: 0°, 45°, 90°, 135°
    kernels = [] # สร้าง list เปล่า
    for theta in thetas: #  0°, 45°, 90°, 135°
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F) # ktype=cv2.CV_32F บอกว่าเป็น float32
        # Gabor kernels ให้ค่าแรงเมื่อเจอลวดลายในภาพที่มี ทิศทางและความถี่สอดคล้อง กับ kernel 
        kernels.append(kernel) # เก็บ kernel ที่สร้างเสร็จลงใน list kernels
    return kernels

def apply_gabor(img, kernels):
    # ประมวลผลด้วยทุก kernel แล้วใช้ max เพื่อตรวจลายเส้นเด่นสุด
    accum = np.zeros_like(img, dtype=np.float32) # np.zeros_like(img) คือ ขนาดจะเท่ากับ img โดยไม่ต้องกำหนด row cols
    for kernel in kernels:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel) # ทำ 2D convolution กับ img โดยใช้ Gabor kernel
        accum = np.maximum(accum, filtered) # เลือกค่าที่สูงที่สุดในแต่ละพิกเซลระหว่าง accum , filtered ของ kernel ปัจจุบัน
    accum = cv2.normalize(accum, None, 0, 255, cv2.NORM_MINMAX)  # ค่าต่ำสุดใน accum  กลายเป็น 0  , ค่าสูงสุดใน accum   กลายเป็น 255 , ที่เหลือถูกกระจาย
    return accum.astype(np.uint8)

# เตรียม kernel
gabor_kernels = build_gabor_kernels()

# เดินทุกโฟลเดอร์
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

                enhanced_img = apply_gabor(img, gabor_kernels)
                cv2.imwrite(output_img_path, enhanced_img)

print("Gabor enhancement เสร็จแล้ว!")
print("เก็บไว้ที่: data/Enhanced-Gabor")
