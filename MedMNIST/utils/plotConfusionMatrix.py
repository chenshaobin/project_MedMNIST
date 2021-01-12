import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# plt.rcParams['savefig.dpi'] = 50
# plt.rcParams['figure.dpi'] = 50

font1 = {'family': 'sans-serif', 'weight': 'normal', 'size': 16}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title('Accuracy = {:.2f}%'.format(accu), font1)
    #
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    # plt.yticks(tick_marks, classes, fontsize=12)

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    #cm = cm*100
    print(cm)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    # plt.title('Accuracy = {:.2f}%'.format(accu), font1)
    plt.title(title, fontsize=10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontdict=font1)

    # cbar = plt.colorbar(plt.imshow(cm, interpolation='nearest', cmap=cmap))
    # cbar.set_clim(vmin=0, vmax=100)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', verticalalignment='top', fontsize=16)
    plt.tight_layout()
    # plt.ylabel('True label', fontsize=12)
    # plt.xlabel('Predicted label', fontsize=12)
