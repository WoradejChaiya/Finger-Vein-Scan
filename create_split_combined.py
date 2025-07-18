import os
import pandas as pd

# โฟลเดอร์หลักที่เก็บภาพที่ผ่านการ enhancement แล้ว
base_dir = "data/Enhanced-Combined"
splits = ["Train", "Val", "Test"]

data = []

# วนลูปชุด Train / Val / Test
for split in splits:
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        continue
    for id_name in os.listdir(split_dir):  # เช่น 10001, 10801
        id_path = os.path.join(split_dir, id_name)
        if not os.path.isdir(id_path):
            continue
        for fname in os.listdir(id_path):
            if fname.endswith(".png"):
                relative_path = os.path.join("data", "Enhanced-Combined", split, id_name, fname)
                data.append({
                    "filepath": relative_path.replace("\\", "/"),
                    "id": id_name,
                    "split": split.lower()
                })

# สร้าง DataFrame
df = pd.DataFrame(data)

# บันทึกเป็น CSV
df.to_csv("split_combined.csv", index=False)
print(f"สร้าง split_combined.csv สำเร็จ: {len(df)} ภาพ จาก {df['id'].nunique()} ID")
