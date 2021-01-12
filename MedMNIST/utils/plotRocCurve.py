
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import os


font1 = {'family': 'sans-serif', 'weight': 'normal', 'size': 16}
def plotRocCurve(data_name, true_label, scores, save_path, picName):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    config = {
        "pathmnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-.', 'k-', 'w-'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',  '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',  '{} (AUC = {:.3f})',]
        },
        "chestmnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-.', 'k-', 'w-', 'g--', 'y--', 'r--', 'b--', 'c--'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "dermamnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4, 5, 6],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-.'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "octmnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3],
            "color_index": ['g-', 'y-', 'r-', 'b-'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "pneumoniamnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2],       # 2 class
            "color_index": ['g-', 'y-'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "retinamnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})']
        },
        "breastmnist": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2],
            "color_index": ['g-', 'y-'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "organmnist_axial": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-.', 'k-', 'w-', 'g--', 'y--'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "organmnist_coronal": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-.', 'k-', 'w-', 'g--', 'y--'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        },
        "organmnist_sagittal": {
            "roc_curve_Name": data_name,
            "class_label": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "color_index": ['g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-.', 'k-', 'w-', 'g--', 'y--'],
            "label_index": ['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})',
                            '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
        }
    }
    args = config[data_name]
    roc_curve_Name = args["roc_curve_Name"]   # 'retinamnist'
    class_label = args["class_label"]      # [0, 1, 2, 3, 4]
    class_number = len(class_label)    # len(class_label), 2
    # Binarize the output 将类别标签二值化
    true_label = label_binarize(true_label, classes=class_label)
    scores = np.array(scores)
    # one vs rest方式计算每个类别的TPR/FPR以及AUC
    for i in range(class_number):
        fpr[i], tpr[i], _ = roc_curve(np.array(true_label[:, i]), scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    color_index = args["color_index"]   # ['g-', 'y-', 'r-', 'b-', 'c-']  # 'g-', 'y-', 'r-', 'b-', 'c-', 'm-', 'y-', 'k-', 'w-'
    label_index = args["label_index"]   #['{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})', '{} (AUC = {:.3f})']
    f = plt.figure(figsize=(10, 10))
    for i in range(class_number):
        plt.plot(fpr[i], tpr[i], color_index[i], lw=2,
                 label=label_index[i].format(class_label[i], roc_auc[i]))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('{} ROC_curve'.format(roc_curve_Name), fontsize=18)
    plt.legend(loc="lower right", prop=font1)
    # Vgg16_rocCurve.png alex_rocCurve.png resnet_rocCurve
    plt.savefig(os.path.join(save_path, picName), dpi=600)
    # plt.show()
    plt.clf()

# 1 place