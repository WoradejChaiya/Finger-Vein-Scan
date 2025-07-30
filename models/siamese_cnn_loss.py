# models/siamese_cnn_loss.py

import torch  # ใช้สำหรับ tensor และคำนวณ
import torch.nn as nn  # โมดูลสำหรับ neural network
import torch.nn.functional as F  # ฟังก์ชันช่วยเช่น pairwise_distance

# นิยาม ContrastiveLoss ในรูปแบบของ nn.Module
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):  # สร้าง class และกำหนด margin (ระยะห่างขั้นต่ำ)
        super(ContrastiveLoss, self).__init__()  # เรียก constructor ของ nn.Module
        self.margin = margin  # บันทึกค่า margin ไว้ใน object

    def forward(self, output1, output2, label):  # ฟังก์ชันหลัก ใช้คำนวณ loss
        euclidean_distance = F.pairwise_distance(output1, output2)  # ระยะห่างแบบ Euclidean ระหว่าง embeddings
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +  # ถ้าต่าง class → รักษาระยะห่างให้มาก
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # ถ้าเหมือนกัน → บีบให้ใกล้
        )
        return loss  # ส่งผลลัพธ์ loss กลับ
