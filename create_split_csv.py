import os  # os – ใช้จัดการกับ path, ไฟล์ และโฟลเดอร์บนเครื่อง
import pandas as pd  # ใช้สร้างและจัดการกับตารางข้อมูล (DataFrame) เพื่อเขียนเป็น .csv


# โฟลเดอร์ที่เก็บ train / val / test
base_dir = "data"  # ชื่อโฟลเดอร์หลักที่เก็บข้อมูล (data/...)
splits = ['Train', 'Val', 'Test']  # โฟลเดอร์ย่อยที่สร้างไว้ใน data/  (พิมพ์เล็กใหญ่ต้องถูก)


# เตรียมเก็บข้อมูล
data = []  # ให้เป็น list ที่ใช้เก็บข้อมูลของแต่ละภาพ เช่น path, id, และ split

for split in splits: 
    split_dir = os.path.join(base_dir, split)  # ใช้รวม path หลายส่วนให้เป็น path เดียว  เช่น 	data/train
    id_dirs = os.listdir(split_dir)  # ดึงรายชื่อไฟล์หรือโฟลเดอร์ใน path 

    for id_name in id_dirs:  # วนลูปแต่ละ ID เช่น 10001
        id_path = os.path.join(split_dir, id_name) # สร้าง path ของโฟลเดอร์ของแต่ละ ID  เช่น split_dir = "data/Train"  id_name = "00001"   ผลลัพท์ id_path = "data/Train/00001"
        if not os.path.isdir(id_path):  # เช็กว่าเป็นโฟลเดอร์ไหม
            continue
        for fname in os.listdir(id_path): # จะคืนชื่อไฟล์ เช่น 10001_001_Shift.png
            if fname.endswith('.png'):  # เช็กว่าไฟล์นั้นเป็น .png ไหม
                relative_path = os.path.join(base_dir, split, id_name, fname) # สร้าง path เช่น data/Train/00001/00001_001_Shift.png
                data.append({ 
                    'filepath': relative_path.replace("\\", "/"), # ใช้โหลดภาพ
                    'id': id_name, # ใช้เป็น label
                    'split': split.lower()  # บอกว่าภาพนี้อยู่ในชุดไหน train / val / test 
                })


# สร้าง DataFrame และบันทึกเป็น CSV
df = pd.DataFrame(data)  #  แปลงข้อมูลสองมิติคือมี columns, rows
df.to_csv("split.csv", index=False)  # บันทึกไฟล์ตารางเป็น split.csv
print(f"✅ สร้าง split.csv สำเร็จ: {len(df)} ภาพ")
