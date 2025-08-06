import os, sys
sys.path.append(os.path.abspath("."))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

def resolve_image_path(row):
    fname = os.path.basename(row["filepath"])
    for root, _, files in os.walk("data"):
        if fname in files:
            return os.path.join(root, fname)
    # ถ้าไม่เจอไฟล์ ให้ return None
    print(f"Warning: ไม่พบไฟล์ {fname} ... ข้าม")
    return None

def main():
    # 1. Import ทุกอย่างที่ต้องใช้
    from datasets.siamese_cnn_dataset import SiameseCnnDataset
    from models.siamese_cnn_model import SiameseCnnModel
    from transforms.transforms_config import transform_pipeline as siamese_transform

    from datasets.vit_dataset import ViTDataset
    from models.vit_model import ViTModel
    from transforms.vit_transforms import vit_transform_pipeline as vit_transform

    from datasets.efficientnet_b0_dataset import EfficientNetB0Dataset
    from models.efficientnet_b0_model import EfficientNetB0Model
    from transforms.efficientnet_b0_transforms import transform_pipeline as eff_transform

    from utils.metrics import accuracy, far_frr_eer

    CSV_PATH = "data/split_combined.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Load CSV + Filter เฉพาะไฟล์ที่หาเจอ
    df = pd.read_csv(CSV_PATH)
    df["filepath"] = df.apply(resolve_image_path, axis=1)
    df = df[df["filepath"].notnull()].reset_index(drop=True)
    unique_ids = sorted(df["id"].unique())
    id_to_idx = {id_: idx for idx, id_ in enumerate(unique_ids)}
    df["label_idx"] = df["id"].map(id_to_idx)

    results = {"model": [], "accuracy": [], "eer": [], "far": [], "frr": []}

    # 3. SiameseCNN
    siam_ds = SiameseCnnDataset(df["filepath"], df["label_idx"], siamese_transform)
    siam_loader = DataLoader(siam_ds, batch_size=64, num_workers=4, pin_memory=True)
    siam_model = SiameseCnnModel().to(DEVICE).eval()
    all_dists, all_labels, all_preds = [], [], []

    with torch.no_grad():
        for img1, img2, lbl in tqdm(siam_loader, desc="SiameseCNN"):
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            emb1, emb2 = siam_model(img1, img2)
            dists = torch.nn.functional.pairwise_distance(emb1, emb2).cpu().numpy()
            preds = (dists < 1.0).astype(int)
            all_dists.extend(dists)
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())

    siam_acc = accuracy(all_labels, all_preds)
    siam_eer, siam_far, siam_frr, _, _ = far_frr_eer(all_dists, all_labels)
    results["model"].append("SiameseCNN")
    results["accuracy"].append(siam_acc)
    results["eer"].append(siam_eer)
    results["far"].append(siam_far)
    results["frr"].append(siam_frr)

    # Save y_true, y_pred, dists สำหรับ Confusion Matrix และ ROC
    with open("results/SiameseCNN_cm_data.pkl", "wb") as f:
        pickle.dump([all_labels, all_preds], f)
    with open("results/SiameseCNN_roc_data.pkl", "wb") as f:
        pickle.dump([all_labels, all_dists], f)

    # 4. ViT & EfficientNet
    for model_name, ModelCls, DatasetCls, trans in [
        ("ViT", ViTModel, ViTDataset, vit_transform),
        ("EfficientNetB0", EfficientNetB0Model, EfficientNetB0Dataset, eff_transform)
    ]:
        ds = DatasetCls(df["filepath"], df["label_idx"], trans)
        loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)

        model = ModelCls(num_classes=len(unique_ids)).to(DEVICE).eval()
        all_preds, all_labels, all_scores = [], [], []

        with torch.no_grad():
            for img, lbl in tqdm(loader, desc=model_name):
                logits = model(img.to(DEVICE))
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbl.numpy())
                scores = torch.softmax(logits, dim=1).max(dim=1).values.cpu().numpy()
                all_scores.extend(scores)

        acc = accuracy(all_labels, all_preds)
        try:
            # ถ้า metrics.far_frr_eer รองaccรับ multiclass
            _, far, frr, _, _ = far_frr_eer(all_preds, all_labels)
        except:
            far, frr = float("nan"), float("nan")
        results["model"].append(model_name)
        results["accuracy"].append(acc)
        results["eer"].append(float("nan"))  # ถ้าไม่ได้คำนวณ
        results["far"].append(far)
        results["frr"].append(frr)

        # Save y_true, y_pred, y_score สำหรับ plot CM/ROC
        with open(f"results/{model_name}_cm_data.pkl", "wb") as f:
            pickle.dump([all_labels, all_preds], f)
        with open(f"results/{model_name}_roc_data.pkl", "wb") as f:
            pickle.dump([all_labels, all_scores], f)

    # 5. Save metrics summary
    os.makedirs("results", exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv("results/metrics.csv", index=False)
    print(df_res)

if __name__ == "__main__":
    main()
