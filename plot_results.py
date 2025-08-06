import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# 1. Load Metrics
metrics = pd.read_csv("results/metrics.csv") # ต้องมี column: model, accuracy, eer, far, frr

# 2. Bar Chart: Accuracy/EER
fig, ax = plt.subplots()
width = 0.35
x = np.arange(len(metrics["model"]))
ax.bar(x - width/2, metrics["accuracy"], width, label="Accuracy")
ax.bar(x + width/2, metrics["eer"], width, label="EER")
ax.set_xticks(x)
ax.set_xticklabels(metrics["model"], rotation=30)
ax.set_title("Comparison of Accuracy and EER")
ax.set_ylabel("Value")
ax.legend()
plt.tight_layout()
plt.savefig("results/accuracy_eer_comparison.png")
plt.close()

# 3. Bar Chart: FAR/FRR
fig, ax = plt.subplots()
ax.bar(x - width/2, metrics["far"], width, label="FAR")
ax.bar(x + width/2, metrics["frr"], width, label="FRR")
ax.set_xticks(x)
ax.set_xticklabels(metrics["model"], rotation=30)
ax.set_title("Comparison of FAR and FRR")
ax.set_ylabel("Value")
ax.legend()
plt.tight_layout()
plt.savefig("results/far_frr_comparison.png")
plt.close()

# 4. ROC Curve (สมมติว่ามีข้อมูล y_true, y_score แยกเซฟไว้ หรือ load ใหม่)
import pickle

for model in metrics["model"]:
    try:
        # ต้องเตรียมให้มี y_true, y_score (หรือ dists) ต่อโมเดล เช่น pickle หรือ npy ไฟล์
        with open(f"results/{model}_roc_data.pkl", "rb") as f:
            y_true, y_score = pickle.load(f)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model} (AUC={roc_auc:.2f})")
    except Exception as e:
        print(f"Skip {model}: {e}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (FAR vs FRR)")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

# 5. Confusion Matrix (load y_true, y_pred ต่อโมเดล)
for model in metrics["model"]:
    try:
        with open(f"results/{model}_cm_data.pkl", "rb") as f:
            y_true, y_pred = pickle.load(f)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.title(f"Confusion Matrix: {model}")
        plt.savefig(f"results/confusion_matrix_{model}.png")
        plt.close()
    except Exception as e:
        print(f"Skip CM for {model}: {e}")

print("Plotting completed! See images in results/")
