import os
import argparse
from tqdm import trange
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from medmnist.models import ResNet18, ResNet50
from medmnist.dataset import PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST, \
    BreastMNIST, OrganMNISTAxial, OrganMNISTCoronal, OrganMNISTSagittal
from medmnist.evaluator import getAUC, getACC, save_results
from medmnist.info import INFO
from utils.plotRocCurve import plotRocCurve
from utils.metric import metric_results, printMetricResults
from utils.plotConfusionMatrix import plot_confusion_matrix
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
class Config():
    confusionMatrixPath = './result/confusionMatrix'
    rocPic_path = './result/RocCurve'

config = Config()
if not os.path.isdir(config.confusionMatrixPath):
    os.makedirs(config.confusionMatrixPath)
if not os.path.isdir(config.rocPic_path):
    os.makedirs(config.rocPic_path)

def main(flag, input_root, output_root, end_epoch, download):
    """
    main function
    :param flag: name of subset

    """

    flag_to_class = {
        "pathmnist": PathMNIST,
        "chestmnist": ChestMNIST,
        "dermamnist": DermaMNIST,
        "octmnist": OCTMNIST,
        "pneumoniamnist": PneumoniaMNIST,
        "retinamnist": RetinaMNIST,
        "breastmnist": BreastMNIST,
        "organmnist_axial": OrganMNISTAxial,
        "organmnist_coronal": OrganMNISTCoronal,
        "organmnist_sagittal": OrganMNISTSagittal,
    }
    DataClass = flag_to_class[flag]

    info = INFO[flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    start_epoch = 0
    lr = 0.001
    batch_size = 128
    val_auc_list = []
    dir_path = os.path.join(output_root, '%s_checkpoints' % (flag))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('==> Preparing data...')
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[.5], std=[.5])])
    # load data
    train_dataset = DataClass(root=input_root, split='train', transform=train_transform,  download=download)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = DataClass(root=input_root, split='val', transform=val_transform, download=download)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = DataClass(root=input_root, split='test', transform=test_transform, download=download)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print('==> Building and training model...')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print("use:", device)
    model = ResNet50(in_channels=n_channels, num_classes=n_classes).to(device)
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, criterion, train_loader, device, task)
        val(model, val_loader, device, val_auc_list, task, dir_path, epoch)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    print('epoch %s is the best model' % index)

    print('==> Testing model...')
    restore_model_path = os.path.join(
        dir_path, 'ckpt_%d_auc_%.5f.pth' % (index, auc_list[index]))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, 'train', train_loader, device, flag, task, output_root=output_root)
    test(model, 'val', val_loader, device, flag, task, output_root=output_root)
    test(model, 'test', test_loader, device, flag, task, output_root=output_root)


def train(model, optimizer, criterion, train_loader, device, task):
    """
    training function
    :param model: the model to train
    :param optimizer: optimizer used in training
    :param criterion: loss function
    :param train_loader: DataLoader of training set
    :param device: cpu or cuda
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class

    """

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long().to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, task, dir_path, epoch):
    """ validation function
    :param model: the model to validate
    :param val_loader: DataLoader of validation set
    :param device: cpu or cuda
    :param val_auc_list: the list to save AUC score of each epoch
    :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
    :param dir_path: where to save model
    :param epoch: current epoch

    """

    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, task)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch,
    }

    path = os.path.join(dir_path, 'ckpt_%d_auc_%.5f.pth' % (epoch, auc))
    torch.save(state, path)


def test(model, split, data_loader, device, flag, task, output_root=None):
    """
        testing function
        :param model: the model to test
        :param split: the data to test, 'train/val/test'
        :param data_loader: DataLoader of data
        :param device: cpu or cuda
        :param flag: subset name
        :param task: task of current dataset, binary-class/multi-class/multi-label, binary-class
        :param data_name: data name
    """
    configuration = {
        "pathmnist": {
            "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        },
        "chestmnist": {
            "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
        },
        "dermamnist": {
            "classes": ['0', '1', '2', '3', '4', '5', '6']
        },
        "octmnist": {
            "classes": ['0', '1', '2', '3']
        },
        "retinamnist": {
            "classes": ['0', '1', '2', '3', '4']
        },
        "pneumoniamnist": {
            "classes": ['0', '1']
        },
        "breastmnist": {
            "classes": ['0', '1']
        },
        "organmnist_axial": {
            "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        },
        "organmnist_coronal": {
            "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        },
        "organmnist_sagittal": {
            "classes": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        }
    }
    args = configuration[flag]
    model.eval()

    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)
    pred_labels = []
    true_labels = []
    scores = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            predict_label = outputs.data.max(1)[1].cpu().numpy()
            pred_labels.extend(predict_label)
            true_label = targets.data.cpu().numpy()
            true_labels.extend(true_label)
            scores.extend(outputs.data.cpu().numpy().tolist())

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        if split == 'test' and flag != 'chestmnist':
            result = metric_results(pred_labels, true_labels)
            printMetricResults(result)
            plot_confusion_matrix(result['confusion_matrix'], classes=args["classes"],
                                  normalize=False,
                                  title=flag)
            confusion_matrixPath = os.path.join(config.confusionMatrixPath, flag)
            plt.savefig(confusion_matrixPath, dpi=600)
            plt.clf()
            rocPicName = flag
            plotRocCurve(flag, true_labels, scores, config.rocPic_path, rocPicName)

        auc = getAUC(y_true, y_score, task)
        acc = getACC(y_true, y_score, task)
        print('%s AUC: %.5f ACC: %.5f' % (split, auc, acc))

        if output_root is not None:
            output_dir = os.path.join(output_root, flag)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_path = os.path.join(output_dir, '%s.csv' % (split))
            save_results(y_true, y_score, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RUN Baseline model of MedMNIST')
    parser.add_argument('--data_name',
                        default='chestmnist',
                        help='subset of MedMNIST',
                        type=str)
    parser.add_argument('--input_root',
                        # default='./input',
                        default='/home/ubuntu/chenshaobin/data/medmnist',
                        help='input root, the source of dataset files',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--num_epoch',
                        default=50,
                        help='num of epochs of training',
                        type=int)
    parser.add_argument('--download',
                        default=False,   # True
                        help='whether download the dataset or not',
                        type=bool)

    args = parser.parse_args()
    data_name = args.data_name.lower()
    input_root = args.input_root
    output_root = args.output_root
    end_epoch = args.num_epoch
    download = args.download
    main(data_name,
         input_root,
         output_root,
         end_epoch=end_epoch,
         download=download)
