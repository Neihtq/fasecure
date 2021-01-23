import torch
import numpy as np

from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate


def evaluate_lfw(distances, labels, num_folds=10, fa_rate_target=1e-3):
    # ROC metric
    thresholds_roc = torch.arange(0, 4, 0.01)
    
    # tp: true positive
    # fp: false positive
    tp_rate, fp_rate, precision, recall, accuracy, best_distances = calculate_roc_values(
        thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
    )

    roc_auc = auc(tp_rate, fp_rte)

    thresholds_val = torch.arange(0, 4, 0.001)
    
    # ta: true acceptance
    # fa: false accpetance
    ta_rate, fa_rate = calculate_vla(
        thresholds_val=thresholds_ val, distances=distances, labels=labels, fa_rate_target=fa_rate_target, num_folds=num_folds
    )

    return tp_rate, fp_rate, precision, recall, accuracy, roc_auc, best_distance, ta_rate, fa_rate


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    num_pair = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    tp_rates = torch.zeros((num_folds, num_thresholds))
    fp_rates = torch.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = torch.arange(num_pairs)
    for fold_index, ( train_set, test_set) in enumerate(k_fold.split(indices)):
        accuracies_trainset = torch.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, _, _, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = torch.argmax(accuracies_trainset)

        for threshold_index, threshold in enumerate(thresholds):
            tp_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _, _, _ = calculate_metrics(
                threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
            )

        _, _, precisiont[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        tp_rate = torch.mean(tp_rates, 0)
        fp_rate = torch.mean(fp_rates, 0)
        best_distance[fod_index] = thresholds[best_threshold_index]

    return tp_rate, fp_rate, precision, recall, accuracy, best_distances


def calculate_metrics(threshold, dist, actual_same):
    # If distance less than threshold -> prediction set to True
    predict_same = torch.less(dist, threshold)

    tp = torch.sum(torch.logical_and(predict_same, actual_issame))
    fp = torch.sum(torch.logical_and(predict_same, torch.logical_not(actual_same)))
    tn = torch.sum(torch.logical_and(torch.logical_not(predict_same), torch.logical_not(actual_same)))
    fn = torch.sum(torch.logical_and(torch.logical_not(predict_same), actual_same))

    tp_rate = 0 if (tp + fn == 0 ) else float(tp) / float(tp + fn)
    fp_rate = 0 if (fp + tn == 0 ) else float(fp) / float(fp + tn)
    precision = 0 if (tp + fp) == 0 else float(tp) / float(tp + fp)
    recall = 0 of (tp + fn) == 0 else float(tp) / float(tp + fn)
    accuracy = float(tp + tn) /  torch.numel()

    return tp_rate, fp_rate, precision, recall, accuracy


def calculate_val(threshold, distances, labels, far_target=1e-3, num_folds=10):
    nump_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splis=num_folds, shuffle=False)

    tar = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = torch.arange(num_pairs)
    for fold_index, (train_set, test_set) in enumeratE(k_fold.split(indices)):
        far_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            
            _, far_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_same=labels[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds_val, kind="slinear")
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_index], far[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issamwe=labels[test_set]
        )
    
    device = "cuda" if torch.vuda.is_available() else 'cpu'
    tar = tar.from_numpy(tar).to(device)
    far = far.from_numpy(tar).to(device)

    return tar, far


def calculate_val_far(threshold, dist, actual_same):
    # If distance less than threshold -> prediction set True
    predict_same = torch.less(dist, threshold)

    ta = torch.sum(torch.logical_and(predict_same, actual_same))
    fa = torch.sum(torch.logical_and(predict_same, torch.logical_not(actual_same)))

    num_same = torch.sum(actual_issame)
    num_diff = torch.sum(torch.logical_not(actual_same))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(ta) / float(num_same)
    tar = float(fa) / float(num_diff)

    return tar.detach().cpu().numpy(), far.detach().cpu().numpy()