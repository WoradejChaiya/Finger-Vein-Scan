import os  # จัดการ path
import sys  # ปรับ sys.path ให้หา src เจอ
import torch  # PyTorch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import โมเดลทั้งสาม
from models.siamese_cnn_model import SiameseCnnModel
from models.vit_model import ViTModel
from models.efficientnet_b0_model import EfficientNetB0Model

def test_siamese():
    """ทดสอบ forward pass ของ Siamese CNN"""
    model = SiameseCnnModel()  # สร้างโมเดล
    model.eval()               # สลับเป็น eval mode
    img1 = torch.randn(4, 1, 224, 224)  # dummy input 4 ชุด
    img2 = torch.randn(4, 1, 224, 224)  # dummy input 4 ชุด
    with torch.no_grad():      # ไม่มี gradient
        emb1, emb2 = model(img1, img2)  # forward pass
    print("SiameseCNN embeddings:", emb1.shape, emb2.shape)  # ควรเป็น [4,128], [4,128]

def test_vit():
    """ทดสอบ forward pass ของ ViT (Tiny)"""
    num_classes = 5  # กำหนด arbitrary classes
    model = ViTModel(num_classes=num_classes)  # สร้างโมเดล
    model.eval()               # eval mode
    x = torch.randn(4, 1, 224, 224)  # dummy input
    with torch.no_grad():
        logits = model(x)      # forward pass
    print("ViT output shape:", logits.shape)  # ควรเป็น [4,5]

def test_efficientnet():
    """ทดสอบ forward pass ของ EfficientNet-B0"""
    num_classes = 5
    model = EfficientNetB0Model(num_classes=num_classes)  # สร้างโมเดล
    model.eval()               # eval mode
    x = torch.randn(4, 1, 224, 224)  # dummy input
    with torch.no_grad():
        logits = model(x)      # forward pass
    print("EfficientNetB0 output shape:", logits.shape)  # ควรเป็น [4,5]

if __name__ == "__main__":
    print("=== Testing forward pass of all models ===")
    test_siamese()        # รัน Siamese CNN
    test_vit()            # รัน ViT
    test_efficientnet()   # รัน EfficientNet-B0
