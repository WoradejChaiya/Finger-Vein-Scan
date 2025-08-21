# plots_report_pro.py
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, det_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ---------- create folders ----------
os.makedirs("results", exist_ok=True)
os.makedirs("results/image", exist_ok=True)  # เก็บภาพทั้งหมดในโฟลเดอร์นี้

# ---------- helpers ----------
def annot_bar(ax, fmt=".2f"):
    fmt = fmt.lstrip(":")
    for p in ax.patches:
        h = float(p.get_height())
        ax.annotate(f"{h:{fmt}}%", (p.get_x() + p.get_width()/2, h),
                    ha="center", va="bottom", fontsize=9, xytext=(0,3),
                    textcoords="offset points")

def far_frr_eer_from_pairs(y_true, y_score):
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]; scores_sorted = y_score[order]
    P = int(np.sum(y_true_sorted == 1)); N = int(np.sum(y_true_sorted == 0))
    tp = fp = 0; best = (1.0, 1.0, 1.0, 1.0, 0.0)
    for s, t in zip(scores_sorted, y_true_sorted):
        if t == 1: tp += 1
        else: fp += 1
        far = fp / N; frr = (P - tp) / P; diff = abs(far - frr)
        if diff < best[0]:
            eer = (far + frr) / 2.0
            best = (diff, eer, far, frr, s)
    _, eer, far, frr, thr = best
    return eer, far, frr, thr

def frr_at_far(y_true, y_score, target_far):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.searchsorted(fpr, target_far, side="left")
    if idx == 0: return fnr[0]
    if idx >= len(fpr): return fnr[-1]
    x0, x1 = fpr[idx-1], fpr[idx]; y0, y1 = fnr[idx-1], fnr[idx]
    t = (target_far - x0) / (x1 - x0 + 1e-12)
    return (1 - t) * y0 + t * y1

def accuracy_curves(y_true, y_score):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    P = (y_true == 1).sum(); N = (y_true == 0).sum(); T = P + N
    tp = tpr * P
    tn = (1 - fpr) * N
    acc = (tp + tn) / T
    precision = tp / (tp + fpr * N + 1e-12)
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return thr, acc, f1

def to_pct(series):
    return series * 100

# ---------- load tables ----------
df_ver   = pd.read_csv("results/metrics_verification.csv")
df_ident = pd.read_csv("results/metrics_identification.csv")
metrics  = (df_ver[["model","eer","far_at_eer","frr_at_eer","acc_at_eer","tn","fp","fn","tp"]]
            .merge(df_ident, on="model", how="outer")
            .rename(columns={"far_at_eer":"far", "frr_at_eer":"frr", "top1_acc":"top1"}))

# ---------- compute extra metrics ----------
extra_rows = []
for model in metrics["model"]:
    try:
        with open(f"results/{model}_roc.pkl","rb") as f:
            y_true, y_score = pickle.load(f)
        y_true = np.asarray(y_true, int); y_score = np.asarray(y_score, float)

        eer, _, _, thr_eer = far_frr_eer_from_pairs(y_true, y_score)
        frr_1e3 = frr_at_far(y_true, y_score, 1e-3)
        frr_1e4 = frr_at_far(y_true, y_score, 1e-4)

        thr, acc_curve, f1_curve = accuracy_curves(y_true, y_score)
        best_idx = int(np.argmax(acc_curve))
        acc_best = float(acc_curve[best_idx]); thr_best = float(thr[best_idx])
        f1_best  = float(f1_curve[best_idx])

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        extra_rows.append({
            "model": model,
            "eer_conf": eer, "thr_eer": thr_eer,
            "frr@far=1e-3": frr_1e3, "frr@far=1e-4": frr_1e4,
            "acc_best": acc_best, "thr_best": thr_best, "f1_best": f1_best,
            "ap": float(ap)
        })

        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"PR Curve: {model} (AP={ap:.2%})")
        plt.tight_layout(); plt.savefig(f"results/image/pr_curve_{model}.png"); plt.close()

    except Exception as e:
        print(f"[warn] skip extra for {model}: {e}")

df_extra = pd.DataFrame(extra_rows)
df_all = metrics.merge(df_extra, on="model", how="left")
df_all.to_csv("results/metrics_with_extra.csv", index=False)
print(df_all.round(6))

# ---------- Confusion Matrix ----------
for _, row in df_all.iterrows():
    try:
        cm = np.array([[row["tn"], row["fp"]],
                       [row["fn"], row["tp"]]], dtype=int)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix: {row['model']}")
        plt.tight_layout()
        plt.savefig(f"results/image/conf_matrix_{row['model']}.png")
        plt.close()
    except Exception as e:
        print(f"[warn] skip CM for {row['model']}: {e}")

# ---------- Plots (ทุกค่าเป็น %) ----------
order = df_all.sort_values("top1", ascending=False)["model"].tolist()
M = df_all.set_index("model").loc[order]
x = np.arange(len(order)); w = 0.38

# (A) Top-1 vs EER
fig, ax = plt.subplots()
ax.bar(x - w/2, to_pct(M["top1"]), width=w, label="Top-1 (%)")
ax.bar(x + w/2, to_pct(M["eer"]),  width=w, label="EER (%)")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25)
ax.set_ylabel("Percentage"); ax.set_title("Identification Top-1 vs Verification EER")
ax.legend(); annot_bar(ax, ".2f")
plt.tight_layout(); plt.savefig("results/image/bar_top1_vs_eer.png"); plt.close()

# (B) FAR / FRR @ EER
fig, ax = plt.subplots()
ax.bar(x - w/2, to_pct(M["far"]), width=w, label="FAR@EER (%)")
ax.bar(x + w/2, to_pct(M["frr"]), width=w, label="FRR@EER (%)")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25)
ax.set_ylabel("Percentage"); ax.set_title("FAR / FRR at EER Threshold")
ax.legend(); annot_bar(ax, ".2f")
plt.tight_layout(); plt.savefig("results/image/bar_far_frr.png"); plt.close()

# (C) FRR at target FAR
fig, ax = plt.subplots()
ax.bar(x - w/2, to_pct(M["frr@far=1e-3"]), width=w, label="FRR @ FAR=0.1%")
ax.bar(x + w/2, to_pct(M["frr@far=1e-4"]), width=w, label="FRR @ FAR=0.01%")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25)
ax.set_ylabel("Percentage"); ax.set_title("Operating Points for Deployment")
ax.legend(); annot_bar(ax, ".2f")
plt.tight_layout(); plt.savefig("results/image/bar_frr_at_target_far.png"); plt.close()

# (D) Accuracy: Best vs @EER
fig, ax = plt.subplots()
ax.bar(x - w/2, to_pct(M["acc_best"]),     width=w, label="Accuracy @ Best thr (%)")
ax.bar(x + w/2, to_pct(M["acc_at_eer"]),   width=w, label="Accuracy @ EER thr (%)")
ax.set_xticks(x); ax.set_xticklabels(order, rotation=25)
ax.set_ylabel("Percentage"); ax.set_title("Verification Accuracy (Best vs EER threshold)")
ax.legend(); annot_bar(ax, ".2f")
plt.tight_layout(); plt.savefig("results/image/bar_acc_best_vs_eer.png"); plt.close()

# (E) ROC (full + zoom)
plt.figure()
for m in order:
    with open(f"results/{m}_roc.pkl","rb") as f: y_true, y_score = pickle.load(f)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr*100, tpr*100, label=f"{m} (AUC={auc(fpr,tpr):.4f})")
plt.plot([0,100],[0,100],"k--"); plt.xlabel("FAR (%)"); plt.ylabel("TPR (%)")
plt.title("ROC Curve (Verification)"); plt.legend()
plt.tight_layout(); plt.savefig("results/image/roc_curve_full.png"); plt.close()

plt.figure()
for m in order:
    with open(f"results/{m}_roc.pkl","rb") as f: y_true, y_score = pickle.load(f)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.plot(fpr*100, tpr*100, label=m)
plt.xlim(0, 1); plt.ylim(95, 100)
plt.xlabel("FAR (%)"); plt.ylabel("TPR (%)"); plt.title("ROC (Zoom FAR≤1%)"); plt.legend()
plt.tight_layout(); plt.savefig("results/image/roc_curve_zoom.png"); plt.close()

# (F) DET curve (linear scale, %)
plt.figure()
for m in order:
    with open(f"results/{m}_roc.pkl","rb") as f: y_true, y_score = pickle.load(f)
    fpr, fnr, _ = det_curve(y_true, y_score)
    plt.plot(fpr*100, fnr*100, label=m)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.xlabel("FAR (%)"); plt.ylabel("FRR (%)"); plt.title("DET Curve (Verification)"); plt.legend()
plt.tight_layout(); plt.savefig("results/image/det_curve_linear_zoom.png"); plt.close()

print("Saved: results/metrics_with_extra.csv + results/image/*.png")
