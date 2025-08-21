# make_gallery_probe.py
import os
import pandas as pd
import numpy as np

# ===== ตั้งค่าพื้นฐาน =====
CSV_PATH = "data/split_combined.csv"  # path ของไฟล์ split_combined.csv
SEED = 42  # seed สำหรับการสุ่ม (ให้ reproducible)
GALLERY_PER_ID = 1  # จำนวนรูปใน gallery ต่อ ID

def main():
    # โหลดข้อมูล
    df = pd.read_csv(CSV_PATH)
    
    # บังคับ id เป็น string
    df['id'] = df['id'].astype(str)
    
    # เลือกเฉพาะ test set
    df_test = df[df['split'] == 'test'].copy()
    print(f"Test set: {len(df_test)} ภาพ, {df_test['id'].nunique()} IDs")
    
    gallery_rows = []
    probe_rows = []
    
    # สุ่มให้ reproducible
    rng = np.random.default_rng(SEED)
    
    # วนตาม ID
    for pid, group in df_test.groupby('id'):
        group = group.sample(frac=1, random_state=SEED)  # shuffle
        
        if len(group) <= GALLERY_PER_ID:
            # ถ้า ID นี้มีรูปน้อยเกินไป ข้าม
            continue
        
        # เลือก GALLERY_PER_ID รูปแรก
        gallery_rows.extend(group.iloc[:GALLERY_PER_ID].to_dict('records'))
        
        # ที่เหลือเป็น probe
        probe_rows.extend(group.iloc[GALLERY_PER_ID:].to_dict('records'))
    
    # สร้าง DataFrame
    df_gallery = pd.DataFrame(gallery_rows)
    df_probe = pd.DataFrame(probe_rows)
    
    # เซฟไฟล์
    os.makedirs("data", exist_ok=True)
    gallery_path = "data/test_gallery.csv"
    probe_path = "data/test_probe.csv"
    df_gallery.to_csv(gallery_path, index=False)
    df_probe.to_csv(probe_path, index=False)
    
    print(f" สร้าง {gallery_path}: {df_gallery.shape[0]} ภาพ, {df_gallery['id'].nunique()} IDs")
    print(f" สร้าง {probe_path}: {df_probe.shape[0]} ภาพ, {df_probe['id'].nunique()} IDs")
    
    # ตรวจสอบภาพซ้ำ
    common = set(df_gallery['filepath']) & set(df_probe['filepath'])
    print(f"ภาพซ้ำระหว่าง gallery/probe: {len(common)}")
    if len(common) > 0:
        print(" มีภาพซ้ำ ตรวจสอบขั้นตอนสุ่ม!")

if __name__ == "__main__":
    main()
