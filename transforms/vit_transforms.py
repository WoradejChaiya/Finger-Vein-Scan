# transforms/vit_transforms.py
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def vit_transform_pipeline(image_size: int = 224, train: bool = True):
    """
    Train: Grayscale(3), Resize(256) -> RandomResizedCrop(224, 0.9–1.0, ratio≈1), Rotate ±5°
    Val/Test: Grayscale(3), Resize(256) -> CenterCrop(224)
    """
    if train:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(int(image_size * 256/224)),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.98, 1.02)),
            T.RandomRotation(degrees=5),
            # T.RandomHorizontalFlip(p=0.1),  # ถ้าดูนิ่งแล้วค่อยเปิด
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return T.Compose([
            T.Grayscale(num_output_channels=3),
            T.Resize(int(image_size * 256/224)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
