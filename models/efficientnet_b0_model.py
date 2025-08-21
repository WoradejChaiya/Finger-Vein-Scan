import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes, pretrained=True,
                 embed_dim=128, l2_norm=True):                      # เพิ่มพารามิเตอร์ฝั่ง embedding
        super(EfficientNetB0Model, self).__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = efficientnet_b0(weights=weights)

        # --- แทนที่ Conv2d ชั้นแรก เพื่อรับ grayscale (1 channel) ---
        old_block = self.model.features[0]       # ConvNormActivation block แรก
        old_conv = old_block[0]                  # Conv2d เดิม (รับ 3 ช่อง)
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        self.model.features[0][0] = new_conv

        # --- ปรับ classifier สุดท้ายให้ตรงกับจำนวนคลาส (ยังคงไว้เพื่อโหลด checkpoint เดิมได้) ---
        in_feats = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_feats, num_classes)

        # --- เลเยอร์ฝั่ง embedding (จาก penultimate features 1280-d → embed_dim) ---
        self.embed_fc = nn.Linear(in_feats, embed_dim)
        self.l2_norm = l2_norm

    # ---- API สำหรับ evaluator: ดึง feature map → GAP → proj → normalize ----
    def extract_features(self, x):
        """
        คืนค่า embedding ขนาด [B, embed_dim] จาก backbone ของ EfficientNet-B0
        """
        y = self.model.features(x)                 # [B, C, H, W]
        y = F.adaptive_avg_pool2d(y, 1).flatten(1) # [B, C]
        z = self.embed_fc(y)                       # [B, embed_dim]
        if self.l2_norm:
            z = F.normalize(z, p=2, dim=1)
        return z

    # ให้ชื่อที่สอดคล้องกับโมเดลอื่น ๆ
    def forward_single(self, x):
        return self.extract_features(x)

    # path ปกติสำหรับงาน classification (ยังใช้ได้ตามเดิม)
    def forward(self, x):
        return self.model(x)
