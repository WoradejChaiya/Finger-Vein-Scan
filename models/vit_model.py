import torch  # เรียกใช้ไลบรารี PyTorch
import torch.nn as nn  # เรียกใช้ฟังก์ชันเกี่ยวกับ neural network ใน PyTorch
from torchvision.models import vit_b_16  # นำเข้า ViT-B/16 (Vision Transformer) จาก torchvision

class ViTModel(nn.Module):  # สร้างคลาส ViTModel ที่สืบทอดจาก nn.Module
    def __init__(self, num_classes=1000, pretrained=True):  # ฟังก์ชันกำหนดค่าเริ่มต้น มีจำนวนคลาสและเลือกใช้ pretrained ได้
        super(ViTModel, self).__init__()  # เรียกใช้งาน constructor ของ nn.Module เดิม
        self.model = vit_b_16(pretrained=pretrained)  # โหลดโมเดล ViT-B/16 แบบ pre-trained

        # ปรับช่อง input grayscale
        self.model.conv_proj = nn.Conv2d(1, 768, kernel_size=(16,16), stride=(16,16))  # เปลี่ยน layer รับภาพ จาก 3 channel (RGB) เป็น 1 channel (grayscale)

        # ปรับ head classifier
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)  # ปรับขนาด fully connected layer สุดท้ายให้ตรงกับจำนวนคลาสที่ต้องการ

    def forward(self, x):  # ฟังก์ชันสำหรับทำ forward pass
        return self.model(x)  # ส่ง input ผ่านโมเดล ViT แล้วคืนค่า output
