import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)


def compute_AUC(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs, AUPRCs of all classes.
    """
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROCs = roc_auc_score(gt_np, pred_np)
        AUPRCs = average_precision_score(gt_np, pred_np)
    except:
        AUROCs = 0.5
        AUPRCs = 0.5

    return AUROCs, AUPRCs


def compute_accuracy(gt, pred):
    return (((pred == gt).sum()) / gt.size(0)).item() * 100


def compute_auprc(gt, pred):
    return average_precision_score(gt, pred)


def compute_accuracy_np_array(gt, pred):
    return np.mean(gt == pred)


def pr_auc(gt, pred, get_all=False):
    precision, recall, _ = precision_recall_curve(gt, pred)
    score = auc(recall, precision)
    if get_all:
        return score, precision, recall
    else:
        return score


# https://www.kaggle.com/code/sohier/probabilistic-f-score
def pfbeta(gt, pred, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(gt)):
        prediction = min(max(pred[idx], 0), 1)
        if (gt[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if c_precision > 0 and c_recall > 0:
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0


def compute_opt_thres(y_true, y_pred, target_fpr=0.15):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        idx_target_fpr_threshold = np.argmin(np.abs(fpr - target_fpr))
        return thresholds[idx_target_fpr_threshold]
    except Exception as e:
        print(f"[compute_opt_thres] Error: {e}. Using threshold=0.5")
        return 0.5

def compute_opt_thres_hard(y_true, y_pred, target_fpr=0.15):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        valid = fpr <= target_fpr
        return thresholds[valid][-1] if np.any(valid) else thresholds[-1]
    except Exception as e:
        print(f"[compute_opt_thres_hard] Error: {e}. Using threshold=0.5")
        return 0.5

def threshold_top_left_roc(y_true, y_pred):
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        best_idx = np.argmin(distances)
        return thresholds[best_idx]
    except Exception as e:
        print(f"[threshold_top_left_roc] Error: {e}. Using threshold=0.5")
        return 0.5

def threshold_for_max_f1(y_true, y_pred):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        precision = precision[:-1]
        recall = recall[:-1]
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        return thresholds[best_idx]
    except Exception as e:
        print(f"[threshold_for_max_f1] Error: {e}. Using threshold=0.5")
        return 0.5

def threshold_for_target_precision(y_true, y_pred, target_precision=0.80):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        precision = precision[:-1]
        valid = precision >= target_precision
        return thresholds[valid][-1] if np.any(valid) else thresholds[-1]
    except Exception as e:
        print(f"[threshold_for_target_precision] Error: {e}. Using threshold=0.5")
        return 0.5

def threshold_for_target_recall(y_true, y_pred, target_recall=0.80):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        recall = recall[:-1]
        valid = recall >= target_recall
        return thresholds[valid][0] if np.any(valid) else thresholds[0]
    except Exception as e:
        print(f"[threshold_for_target_recall] Error: {e}. Using threshold=0.5")
        return 0.5


def all_classification_metrics(gt, pred, target_fpr=0.15):
    # Apply threshold to convert probabilities to class predictions
    threshold_pr_15_soft = compute_opt_thres(gt, pred, target_fpr=0.15)
    threshold_pr_10_soft = compute_opt_thres(gt, pred, target_fpr=0.10)
    threshold_top_left = threshold_top_left_roc(gt, pred)
    threshold_pr_15_hard = compute_opt_thres_hard(gt, pred, target_fpr=0.15)
    threshold_max_f1 = threshold_for_max_f1(gt, pred)
    threshold_target_precision = threshold_for_target_precision(gt, pred)
    threshold_target_recall = threshold_for_target_recall(gt, pred)

    print(f"==========================>>>>> Thresholds <<<<<==========================")
    print(f"threshold_pr_15_soft: {threshold_pr_15_soft}")
    print(f"threshold_pr_10_soft: {threshold_pr_10_soft}")
    print(f"threshold_pr_15_hard: {threshold_pr_15_hard}")
    print(f"threshold_top_left: {threshold_top_left}")
    print(f"threshold_max_f1: {threshold_max_f1}")
    print(f"threshold_target_precision: {threshold_target_precision}")
    print(f"threshold_target_recall: {threshold_target_recall}")
    print(f"==========================>>>>> Thresholds <<<<<==========================")

    pred_label_pr_15_soft = (pred >= threshold_pr_15_soft).astype(int)
    pred_label_pr_10_soft = (pred >= threshold_pr_10_soft).astype(int)
    pred_label_pr_15_hard = (pred >= threshold_pr_15_hard).astype(int)
    pred_label_top_left = (pred >= threshold_top_left).astype(int)
    pred_label_max_f1 = (pred >= threshold_max_f1).astype(int)
    pred_label_target_precision = (pred >= threshold_target_precision).astype(int)
    pred_label_target_recall = (pred >= threshold_target_recall).astype(int)

    metrics = {
        "1a. Accuracy_pr_15_soft": accuracy_score(gt, pred_label_pr_15_soft),
        "1b. Precision_pr_15_soft": precision_score(gt, pred_label_pr_15_soft, zero_division=0),
        "1c. Recall_pr_15_soft": recall_score(gt, pred_label_pr_15_soft, zero_division=0),
        "1d. F1 Score_pr_15_soft": f1_score(gt, pred_label_pr_15_soft, zero_division=0),

        "2a. Accuracy_pr_10_soft": accuracy_score(gt, pred_label_pr_10_soft),
        "2b. Precision_pr_10_soft": precision_score(gt, pred_label_pr_10_soft, zero_division=0),
        "2c. Recall_pr_10_soft": recall_score(gt, pred_label_pr_10_soft, zero_division=0),
        "2d. F1 Score_pr_10_soft": f1_score(gt, pred_label_pr_10_soft, zero_division=0),

        "3a. Accuracy_pr_15_hard": accuracy_score(gt, pred_label_pr_15_hard),
        "3b. Precision_pr_15_hard": precision_score(gt, pred_label_pr_15_hard, zero_division=0),
        "3c. Recall_pr_15_hard": recall_score(gt, pred_label_pr_15_hard, zero_division=0),
        "3d. F1 Score_pr_15_hard": f1_score(gt, pred_label_pr_15_hard, zero_division=0),

        "4a. Accuracy_top_left": accuracy_score(gt, pred_label_top_left),
        "4b. Precision_top_left": precision_score(gt, pred_label_top_left, zero_division=0),
        "4c. Recall_top_left": recall_score(gt, pred_label_top_left, zero_division=0),
        "4d. F1 Score_top_left": f1_score(gt, pred_label_top_left, zero_division=0),

        "5a. Accuracy_max_f1": accuracy_score(gt, pred_label_max_f1),
        "5b. Precision_max_f1": precision_score(gt, pred_label_max_f1, zero_division=0),
        "5c. Recall_max_f1": recall_score(gt, pred_label_max_f1, zero_division=0),
        "5d. F1 Score_max_f1": f1_score(gt, pred_label_max_f1, zero_division=0),

        "6a. Accuracy_target_precision": accuracy_score(gt, pred_label_target_precision),
        "6b. Precision_target_precision": precision_score(gt, pred_label_target_precision, zero_division=0),
        "6c. Recall_target_precision": recall_score(gt, pred_label_target_precision, zero_division=0),
        "6d. F1 Score_target_precision": f1_score(gt, pred_label_target_precision, zero_division=0),

        "7a. Accuracy_target_recall": accuracy_score(gt, pred_label_target_recall),
        "7b. Precision_target_recall": precision_score(gt, pred_label_target_recall, zero_division=0),
        "7c. Recall_target_recall": recall_score(gt, pred_label_target_recall, zero_division=0),
        "7d. F1 Score_target_recall": f1_score(gt, pred_label_target_recall, zero_division=0),

        "AUROC": roc_auc_score(gt, pred),
        "AUPRC": average_precision_score(gt, pred)
    }

    return metrics


def auroc(gt, pred):
    return roc_auc_score(gt, pred)


def pfbeta_binarized(gt, pred):
    positives = pred[gt == 1]
    scores = []
    for th in positives:
        binarized = (pred >= th).astype('int')
        score = pfbeta(gt, binarized, 1)
        scores.append(score)

    return np.max(scores)
