"""
Utility functions
"""

import numpy as np
from sklearn.metrics import *
from sklearn.utils import multiclass
from scipy.stats import pearsonr, spearmanr, kendalltau

import torch


def count_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sum_ep_time(s_time, e_time):
    _time = e_time - s_time
    _mins = int(_time / 60)
    _secs = int(_time % 60)
    return _mins, _secs


def scores_to_metrics(scores, targets, label):
    assert isinstance(scores, torch.Tensor) and isinstance(targets, torch.Tensor)
    assert label in {'gender', 'age', 'income'}
    _scores = scores.detach().cpu().clone().numpy()
    _targets = targets.detach().cpu().clone().numpy()
    metrics = []
    if label == 'gender':  # Binary classification
        raw_metrics = skl_cls_metrics_bin(scores=_scores, y_true=_targets)
        (mean_acc_default, _), pr_auc, (_, f1_max, _), (mcc_default, _) = raw_metrics
        metrics.extend([mean_acc_default, pr_auc, f1_max, mcc_default])
    elif label == 'income':  # Multi-class classification
        raw_metrics = skl_cls_metrics_mul(scores=_scores, y_true=_targets, labels=list(range(1, 10)))
        mean_acc, micro_f1, macro_f1, _, cohen_kappa = raw_metrics
        metrics.extend([mean_acc, macro_f1, micro_f1, cohen_kappa])
    elif label == 'age':  # Regression
        raw_metrics = skl_reg_metrics(scores=_scores, y_true=_targets)
        r2, mae, rmse, ((pr, _), (sr, _), _) = raw_metrics
        metrics.extend([r2, mae, rmse, pr, sr])
    return metrics


def skl_cls_metrics_bin(scores, y_true):
    assert isinstance(scores, np.ndarray) and scores.shape[1] == 2
    assert isinstance(y_true, np.ndarray) and len(y_true.shape) == 1
    assert scores.shape[0] == y_true.shape[0]
    preds_default = np.argmax(scores, axis=1)
    ''' Mean Acc (default) '''
    mean_acc_default = accuracy_score(y_true=y_true, y_pred=preds_default, normalize=True)
    ''' PR_AUC '''
    pr_auc = average_precision_score(y_true=y_true, y_score=scores[:, 1])
    ''' F1 '''
    f1_default = f1_score(y_true=y_true, y_pred=preds_default, average='binary')
    _pres, _recs, _ths = precision_recall_curve(y_true=y_true, probas_pred=scores[:, 1])
    recs, pres, ths = _recs[::-1], _pres[::-1], _ths[::-1]
    assert recs.shape == pres.shape and recs.shape[0] == ths.shape[0] + 1
    f1s = 2 * recs * pres / (recs + pres + 1e-9)  # Numerical stability
    f1_max, f1_th = np.max(f1s), ths[np.argmax(f1s) - 1]  # Max F1 score and corresponding prob threshold
    preds_best = [1 if s >= f1_th else 0 for s in scores[:, 1]]
    ''' Mean Acc (best) '''
    mean_acc_best = accuracy_score(y_true=y_true, y_pred=preds_best, normalize=True)
    ''' Matthews correlation coefficient (MCC) '''
    mcc_default = matthews_corrcoef(y_true=y_true, y_pred=preds_default)
    mcc_best = matthews_corrcoef(y_true=y_true, y_pred=preds_best)
    return (mean_acc_default, mean_acc_best), pr_auc, (f1_default, f1_max, f1_th), (mcc_default, mcc_best)


def skl_cls_metrics_mul(scores, y_true, labels=None):
    assert isinstance(scores, np.ndarray) and scores.shape[1] > 2
    assert isinstance(y_true, np.ndarray) and len(y_true.shape) == 1
    assert scores.shape[0] == y_true.shape[0]
    preds = np.argmax(scores, axis=1)
    if labels is not None:
        data_labels = multiclass.unique_labels(y_true, preds)
        # assert set(labels).issubset(data_labels)
    ''' Mean Acc '''
    mean_acc = accuracy_score(y_true=y_true, y_pred=preds)
    ''' F1 micro macro'''
    micro_f1 = f1_score(y_true=y_true, y_pred=preds, labels=labels, average='micro')
    macro_f1 = f1_score(y_true=y_true, y_pred=preds, labels=labels, average='macro')
    weighted_f1 = f1_score(y_true=y_true, y_pred=preds, labels=labels, average='weighted')
    ''' Matthews correlation coefficient (MCC) '''
    # cohen_kappa = cohen_kappa_score(y1=y_true, y2=preds, labels=labels)
    cohen_kappa = cohen_kappa_score(y1=y_true, y2=preds)
    return mean_acc, micro_f1, macro_f1, weighted_f1, cohen_kappa


def skl_reg_metrics(scores, y_true):
    assert isinstance(scores, np.ndarray) and scores.shape[1] == 1
    assert isinstance(y_true, np.ndarray) and len(y_true.shape) == 1
    assert scores.shape[0] == y_true.shape[0]
    scores = scores[:, 0]  # Flat predicted scores
    ''' R^2 '''
    r2 = r2_score(y_true=y_true, y_pred=scores)
    ''' MAE '''
    mae = mean_absolute_error(y_true=y_true, y_pred=scores)
    ''' RMSE '''
    rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=scores))
    ''' Pearson correlation coefficient and the p-value '''
    pr, pr_p = pearsonr(x=y_true, y=scores)
    ''' Spearman rank-order correlation coefficient and the p-value '''
    sr, sr_p = spearmanr(a=y_true, b=scores, axis=0)
    ''' Kendall’s tau '''
    ktau, ktau_p = kendalltau(x=y_true, y=scores)
    return r2, mae, rmse, ((pr, pr_p), (sr, sr_p), (ktau, ktau_p))
