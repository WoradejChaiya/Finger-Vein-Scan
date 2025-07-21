import torch
import torch.nn.functional as F

def siamese_cnn_contrastive_loss(emb1, emb2, label, margin=1.0): # ฟังก์ชัน loss สำหรับ Siamese Network แบบ contrastive
    euclidean_distance = F.pairwise_distance(emb1, emb2) # คำนวณระยะห่างแบบ Euclidean ระหว่าง embedding ทั้งสอง
    loss = torch.mean( # คำนวณ loss ทั้งหมด แล้วเอาค่าเฉลี่ย
        label * torch.pow(euclidean_distance, 2) +
        (1 - label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    ) 
    # ถ้า label=1 loss = dist²     ถ้า label=0 loss = (margin - dist)²     
    return loss
