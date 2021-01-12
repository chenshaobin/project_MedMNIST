from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def classifier_eval_metric(pred_label, gt_label, pred_score):
    result = {}
    cm = confusion_matrix(gt_label, pred_label)
    result.update({'confusion_matrix': cm})

    accuracy = (cm[0][0] + cm[1][1]) / (cm.sum())
    result.update({'accuracy': accuracy})

    specificity = (cm[0][0]) / (cm[0][1] + cm[0][0])
    result.update({'specificity': specificity})

    precision = precision_score(gt_label, pred_label)
    result.update({'precision': precision})

    recall = recall_score(gt_label, pred_label)
    result.update({'sensitivity': recall})

    F1_score = f1_score(gt_label, pred_label)
    result.update({'F1_score': F1_score})

    # enc = OneHotEncoder()
    # enc.fit(gt_label)
    # targets = enc.transform(gt_label).toarray()
    # pred_score = np.squeeze(np.array(pred_score))
    #
    # AUC = roc_auc_score(targets, pred_score, average='micro')
    targets = np.squeeze(np.array(gt_label))
    pred_score = np.squeeze(np.array(pred_score))
    pred_score = pred_score[:, 1]
    fpr, tpr, threshold = roc_curve(targets, pred_score)
    AUC = auc(fpr, tpr)
    result.update({'AUC': AUC})

    return result


def metric_results(pred_label, gt_label):
    # 计算一些评价标准
    result = {}
    cm = confusion_matrix(gt_label, pred_label)
    result.update({'confusion_matrix': cm})

    precision = precision_score(gt_label, pred_label, average='weighted')
    result.update({'precision': precision})

    accuracy = accuracy_score(gt_label, pred_label)
    result.update({'accuracy': accuracy})

    recall = recall_score(gt_label, pred_label, average='weighted')
    result.update({'recall': recall})  # recall和sensitivity计算方法一样

    F1_score = f1_score(gt_label, pred_label, average='weighted')
    result.update({'F1_score': F1_score})
    return result


def printMetricResults(myDict):
    for item in myDict:
        print(item, ":", '\n', myDict[item])
