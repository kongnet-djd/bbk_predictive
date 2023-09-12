import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

def compute_acc(classif_preds, classif_gts, length):
    classif_preds[classif_preds >= 0.5] = 1.0
    classif_preds[classif_preds < 0.5] = 0
    correct = torch.eq(classif_preds, classif_gts).sum().float().item()
    acc = correct / length
    return acc


def compute_auc(classif_preds, classif_gts):
    '''
    Compute AUC classification score.
    '''
    # classif_preds[classif_preds >= 0.5] = 1.0
    # classif_preds[classif_preds < 0.5] = 0

    auc = roc_auc_score(classif_gts.cpu().detach().numpy(), classif_preds.cpu().detach().numpy())

    return auc

def compute_sensitivity_specificity(classif_preds, classif_gts):
    # 首先，你需要将预测转化为二值结果。假设你的阈值是0.5
    # classif_preds = np.array(classif_preds, dtype=np.float64)
    # classif_gts = np.array(classif_gts, dtype=np.float64)
    binarized_preds = np.float64(classif_preds > 0.5)

    # 然后计算混淆矩阵
    # tn, fp, fn, tp = confusion_matrix(classif_gts.cpu().detach().numpy(), binarized_preds.cpu().detach().numpy()).ravel()
    tn, fp, fn, tp = confusion_matrix(classif_gts, binarized_preds).ravel()

    # 然后，根据混淆矩阵的结果计算灵敏度和特异度
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity
