# models/efficientnet_b0_model.py

import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        # โหลด EfficientNet-B0 พร้อม pretrained weights
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b0(weights=weights)  # backbone ที่ใช้

        # ———— แทนที่ Conv2d ชั้นแรก เพื่อรับ grayscale (1 channel) ————
        # โครงสร้าง features[0] เป็น ConvNormActivation ซึ่งเป็น nn.Sequential
        # ชั้น Conv2d ดั้งเดิมจะอยู่ที่ features[0][0]
        old_block = self.model.features[0]      # ConvNormActivation block แรก
        old_conv = old_block[0]                 # Conv2d ดั้งเดิม
        # สร้าง Conv2d ใหม่ ที่รับ in_channels=1 (grayscale)
        new_conv = nn.Conv2d(
            in_channels=1,                      # เปลี่ยนจาก 3 → 1 channel
            out_channels=old_conv.out_channels, # ตามเดิม
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        # ใส่ Conv2d ใหม่ลงไปแทน block แรก
        self.model.features[0][0] = new_conv

        # ———— ปรับ classifier สุดท้ายให้ตรงกับจำนวนคลาส ————
        in_feats = self.model.classifier[1].in_features  # จำนวนอินพุตเดิมของ head
        self.model.classifier[1] = nn.Linear(in_feats, num_classes)  # วาง Linear ใหม่

    def forward(self, x):
        return self.model(x)  # ส่ง x ผ่านโมเดลทั้งหมด แล้วคืน logits
