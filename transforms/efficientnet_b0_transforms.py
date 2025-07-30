from torchvision import transforms

transform_pipeline = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 1-channel
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                       # -> [1,224,224]
    transforms.Normalize(                        # ปรับ mean/std ให้เป็นของ 1-channel
        mean=[0.5],
        std=[0.5]
    ),
])
