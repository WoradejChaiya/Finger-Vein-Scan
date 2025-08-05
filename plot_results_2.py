# plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

os.makedirs("results/plots", exist_ok=True)

# Accuracy vs EER
plt.figure(figsize=(8,5))
plt.bar(df["model"], df["accuracy"], alpha=0.7, label="Accuracy")
plt.bar(df["model"], df["eer"], alpha=0.7, label="EER")
plt.ylabel("Metric Value")
plt.title("Accuracy vs EER Comparison")
plt.legend()
plt.savefig("results/plots/accuracy_eer_comparison.png")
plt.close()

print("บันทึกกราฟเรียบร้อยที่ results/plots/")
