import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.exceptions import UndefinedMetricWarning


def compute_eval_metrics(gt_labels, pred_labels, probs_pos_cls=None):
    """
    Please see details on how scikit-learn implements "Classification metrics", specifically for
    precision-recall curve
    http://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics

    :param gt_labels:
    :param pred_labels:
    :param probs_pos_cls: 1D numpy array. Assuming probs indicate the prob for positive class
    :return:
    """
    import warnings
    if gt_labels.ndim > 1:
        gt_labels = gt_labels.flatten()
    if probs_pos_cls.ndim > 1:
        probs_pos_cls = probs_pos_cls.flatten()
    if pred_labels.ndim > 1:
        pred_labels = pred_labels.flatten()

    # In case we have no TP (degenerate label=1) or no TN (degenerate label 0) we can't compute AUC measure
    if np.sum(gt_labels) != 0 or np.sum(gt_labels) == gt_labels.shape[0]:
        if probs_pos_cls is not None:
            fpr, tpr, thresholds = roc_curve(gt_labels, probs_pos_cls)
            precision_curve, recall_curve, thresholds = precision_recall_curve(gt_labels, probs_pos_cls)
            roc_auc = roc_auc_score(gt_labels, probs_pos_cls)
            pr_auc = average_precision_score(gt_labels, probs_pos_cls)

        else:
            fpr, tpr, thresholds = roc_curve(gt_labels, pred_labels)
            precision_curve, recall_curve, thresholds = precision_recall_curve(gt_labels, pred_labels)
            roc_auc = roc_auc_score(gt_labels, pred_labels)
            pr_auc = average_precision_score(gt_labels, pred_labels)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                prec, rec, f1, _ = precision_recall_fscore_support(gt_labels, pred_labels, beta=1, labels=1,
                                                                   average="binary")
        except UndefinedMetricWarning:
            print(("WARNING - UndefinedMetricWarning - sum(pred_labels)={}".format(np.sum(pred_labels))))
    else:
        roc_auc = 0
        pr_auc = 0
        f1 = -1.
        prec = -1
        rec = -1
        fpr, tpr, precision_curve, recall_curve = None, None, None, None
    # acc = accuracy_score(gt_labels, pred_labels)

    return f1, roc_auc, pr_auc, prec, rec, fpr, tpr, precision_curve, recall_curve
