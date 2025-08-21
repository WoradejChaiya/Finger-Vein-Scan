import torch.nn as nn  # เรียกใช้โมดูล neural network (nn) ของ PyTorch

vit_loss_fn = nn.CrossEntropyLoss()  # สร้างอ็อบเจกต์ loss function แบบ CrossEntropy สำหรับ classification
