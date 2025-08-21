import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # เพิ่ม tqdm สำหรับ progress bar
import pandas as pd

from datasets.efficientnet_b0_dataset import EfficientNetB0Dataset
from models.efficientnet_b0_model import EfficientNetB0Model
from models.efficientnet_b0_loss import efficientnet_b0_loss_fn
from transforms.efficientnet_b0_transforms import transform_pipeline

# ====== เตรียมข้อมูลจาก CSV ======
df = pd.read_csv('data/split_combined.csv')
train_df = df[df['split'] == 'train']
image_paths = train_df['filepath'].tolist()
id_list = train_df['id'].tolist()

unique_ids = sorted(list(set(id_list)))
id2idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
labels = [id2idx[x] for x in id_list]
num_classes = len(unique_ids)

import pickle
with open("results/efficientnet_b0_unique_ids.pkl", "wb") as f:
    pickle.dump(unique_ids, f)

dataset = EfficientNetB0Dataset(image_paths, labels, transform_pipeline)

# *** Windows: num_workers=0 แก้ bug exited unexpectedly ***
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=12)

# ====== เตรียมโมเดล ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetB0Model(num_classes=num_classes).to(device)

# ====== Loss, Optimizer, Scheduler ======
criterion = efficientnet_b0_loss_fn
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

# ====== Training Loop ======
num_epochs = 5
best_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # tqdm progress bar 
    for images, targets in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "results/efficientnet_b0_best.pth")
        print("Saved new best model.")
