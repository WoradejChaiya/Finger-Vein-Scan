# models/siamese_cnn_model.py
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SiameseCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # [B,512,1,1]
        self.embedding = nn.Linear(512, 128)  # เลเยอร์ที่เทรนด้วย contrastive

    def forward_single(self, x):
        x = self.feature_extractor(x).view(x.size(0), -1)  # [B,512]
        x = self.embedding(x)                              # [B,128]  ← ใช้ของที่เทรนไว้
        return F.normalize(x, p=2, dim=1)                  # ให้พร้อม cosine

    def forward_once(self, x):  # เผื่อโค้ดอื่นเรียกชื่อเดิม
        return self.forward_single(x)

    def forward(self, img1, img2):
        return self.forward_single(img1), self.forward_single(img2)
