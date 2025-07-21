import torch.nn as nn # import โมดูล neural network จาก PyTorch
from torchvision.models import resnet18 # import โมเดล ResNet-18 จาก torchvision

class SiameseCnnModel(nn.Module): # สร้างคลาสโมเดล Siamese ที่สืบทอดจาก nn.Module
    def __init__(self): # ฟังก์ชัน constructor สำหรับกำหนด layers ต่าง ๆ
        super(SiameseCnnModel, self).__init__()  # เรียก constructor ของคลาสแม่ nn.Module
        self.backbone = resnet18(pretrained=True) # โหลดโมเดล ResNet-18 ที่ pretrained บน ImageNet
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # แก้ conv1 ให้รองรับ grayscale image (1 channel)
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])  # เอาเฉพาะ layers ย่อยทั้งหมดของ ResNet ยกเว้น layer สุดท้าย 
        self.embedding = nn.Linear(512, 128) # แปลง feature 512 เป็น 128 

    def forward_once(self, x):
        x = self.feature_extractor(x) # ผ่าน backbone เพื่อดึง feature map
        x = x.view(x.size(0), -1)  # flatten tensor เป็น [batch_size, 512]
        embedding = self.embedding(x) # ผ่าน linear layer เพื่อได้ embedding ขนาด 128
        return embedding # ส่งคืน embedding 

    def forward(self, img1, img2): # รับภาพสองภาพเป็น input
        emb1 = self.forward_once(img1) # ดึง embedding ของภาพที่ 1
        emb2 = self.forward_once(img2) # ดึง embedding ของภาพที่ 2
        return emb1, emb2
