# utils/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix as sk_confusion

def accuracy(y_true, y_pred):
    """คำนวณ Accuracy (ความแม่นยำ)"""
    return accuracy_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    """คำนวณ confusion matrix (TP, TN, FP, FN)"""
    tn, fp, fn, tp = sk_confusion(y_true, y_pred).ravel()
    return tp, tn, fp, fn

def far_frr_eer(distances, labels):
    """คำนวณ FAR, FRR และ EER จาก distance และ labels"""
    thresholds = np.linspace(min(distances), max(distances), 1000)
    fars, frrs = [], []
    for t in thresholds:
        preds = distances <= t
        tp, tn, fp, fn = confusion_matrix(labels, preds)
        far = fp / (fp + tn)
        frr = fn / (fn + tp)
        fars.append(far)
        frrs.append(frr)
    fars, frrs = np.array(fars), np.array(frrs)
    idx_eer = np.nanargmin(np.abs(fars - frrs))
    eer = (fars[idx_eer] + frrs[idx_eer]) / 2
    return eer, thresholds[idx_eer], fars, frrs, thresholds
