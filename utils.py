
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score
import numpy as np  
import os
import torch
from albumentations import Compose, RandomRotate90, Flip, Transpose, \
    OneOf, IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, \
        Blur, ShiftScaleRotate, OpticalDistortion, GridDistortion, IAAPiecewiseAffine, \
            CLAHE, IAASharpen, IAAEmboss, RandomBrightnessContrast, \
                HueSaturationValue, Resize


def data_augmentadion_localizer(p=0.5, train=True):
    if train:
        T =  Compose([
            Resize(height=224, width=224, p=1),
            RandomRotate90(p=p),
            Flip(p=p),
            Transpose(p=p),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2*p),
            OneOf([
                MotionBlur(p=0.2*p),
                MedianBlur(blur_limit=3, p=0.1*p),
                Blur(blur_limit=3, p=0.1*p),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2*p),
            OneOf([
                OpticalDistortion(p=0.3*p),
                GridDistortion(p=0.1*p),
                IAAPiecewiseAffine(p=0.3*p),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3*p),
            HueSaturationValue(p=0.3*p),
        ], p=1) 
    else:
        T = Compose([
                Resize(height=224, width=224, p=1),
        ])
    return T

def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d **函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N

class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

    def write(self, message):
        print(message.rstrip("\n"))
        with open(self.log_path, "a") as f:
            f.write(message + "\n")



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(predictions, labels, scores):
    # predictions = np.array(predictions)
    # labels = np.array(labels)
    # print(predictions.shape)
    # print(labels.shape)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="macro")
    f1 = f1_score(labels, predictions, average="macro")
    sensitivity = sensitivity_score(labels, predictions, average="macro")
    specificity = specificity_score(labels, predictions, average="macro")
    # raise
    if np.array(scores).shape[1] == 2:
        auc = roc_auc_score(labels, np.array(scores)[:, 1], average="macro", multi_class="ovr")
    else:
        auc = roc_auc_score(labels, scores, average="macro", multi_class="ovr")

    mean = np.mean([auc, accuracy, precision, specificity, sensitivity, f1])

    return auc, accuracy, precision, specificity, sensitivity, f1, mean