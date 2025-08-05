# plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
# ที่เก็บไฟล์ metrics ที่บันทึกไว้ (ต้องมีไฟล์ results/metrics.csv)
RESULTS_CSV = os.path.join("results", "metrics.csv")
# โฟลเดอร์สำหรับบันทึกภาพกราฟ
PLOTS_DIR = os.path.join("results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# === โหลด metrics summary ===
df = pd.read_csv(RESULTS_CSV)
# คอลัมน์ที่ต้องมี: model, accuracy, eer

# --- Bar Chart: Accuracy vs EER ---
models = df['model'].tolist()
accuracies = df['accuracy'].tolist()
eers = df['eer'].tolist()

x = range(len(models))
fig, ax = plt.subplots()
ax.bar([i - 0.2 for i in x], accuracies, width=0.4, label='Accuracy')
ax.bar([i + 0.2 for i in x], eers,       width=0.4, label='EER')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Value')
ax.set_title('Comparison of Accuracy and EER')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
bar_path = os.path.join(PLOTS_DIR, 'accuracy_eer_comparison.png')
plt.savefig(bar_path)
plt.close()
print(f"Saved bar chart to {bar_path}")

# --- ROC Curve: FAR vs FRR for each model ---
plt.figure()
for model_name in models:
    # คาดว่ามีไฟล์ results/{model}_fars.csv, frrs.csv
    fars_file = os.path.join('results', f'{model_name}_fars.csv')
    frrs_file = os.path.join('results', f'{model_name}_frrs.csv')
    if os.path.exists(fars_file) and os.path.exists(frrs_file):
        fars = pd.read_csv(fars_file)['far'].values
        frrs = pd.read_csv(frrs_file)['frr'].values
        plt.plot(fars, frrs, marker='o', label=model_name)
    else:
        print(f"Warning: FAR/FRR files not found for {model_name}, skipping ROC")

plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('False Rejection Rate (FRR)')
plt.title('ROC Curve (FAR vs FRR)')
plt.legend()
plt.grid(True)
roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
plt.savefig(roc_path)
plt.close()
print(f"Saved ROC curve to {roc_path}")
