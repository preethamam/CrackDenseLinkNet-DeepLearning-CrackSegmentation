# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:16:13 2019

@author: vmani
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from numba import jit
from natsort import natsorted
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc


def get_scores(TN, FP, FN, TP):
    sensitivity = TP / (TP + FN)
    print("Sensitivity: ", round(sensitivity, 4))

    specificity = TN / (TN + FP)
    print("Specificity: ", round(specificity, 4))

    error_rate = (FP + FN) / (TP + FN + TN + FP)
    print("Error_rate: ", round(error_rate, 4))

    accuracy = 1 - error_rate
    print("Accuracy: ", round(accuracy, 4))

    precision = TP / (TP + FP)
    print("Precision: ", round(precision, 4))

    dice_coeff = (2 * TP) / (2 * TP + FP + FN)
    print("Dice_Coeff: ", round(dice_coeff, 4))

    jaccard = dice_coeff / (2 - dice_coeff)
    print("Jaccard: ", round(jaccard, 4))

    f_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    print("F_score: ", round(f_score, 4), "\n")


@jit(nopython=True)
def confusion_matrix(gnd, pre, TN, FP, FN, TP):
    # print(gnd.shape, pre.shape)
    assert gnd.shape == pre.shape

    for x in range(gnd.shape[0]):

        for y in range(gnd.shape[1]):

            if gnd[x][y] == pre[x][y] == 0:
                TN += 1

            elif gnd[x][y] == pre[x][y] == 255:
                TP += 1

            elif gnd[x][y] == 0 and pre[x][y] == 255:
                FP += 1

            elif gnd[x][y] == 255 and pre[x][y] == 0:
                FN += 1

    # print(TN, FP, FN, TP)
    return TN, FP, FN, TP


def compute_scores(gndPath, predPath):
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    gnd_ids = natsorted(next(os.walk(gndPath))[2])

    pred_ids = natsorted(next(os.walk(predPath))[2])

    assert len(gnd_ids) == len(pred_ids)

    print("Length: ", len(gnd_ids))

    for i in range(len(gnd_ids)):
        
        gnd = cv2.imread(os.path.join(gndPath, gnd_ids[i]), 0)

        pre = cv2.imread(os.path.join(predPath, pred_ids[i]), 0)

        pre[pre < 127.5] = 0
        pre[pre >= 127.5] = 255

        # gnd = cv2.resize(gnd, (pre.shape[1], pre.shape[0]), cv2.INTER_LINEAR)

        # cv2.imwrite(os.path.join(gndPath, gnd_ids[i]), gnd)

        TN, FP, FN, TP = confusion_matrix(gnd, pre, TN, FP, FN, TP)

    return TN, FP, FN, TP


@jit(parallel=True)
def get_roc_plots(gndCupPath, predCupPath, gndDiscPath, predDiscPath):
    gndC_ids = natsorted(next(os.walk(gndCupPath))[2])
    predC_ids = natsorted(next(os.walk(predCupPath))[2])

    assert len(gndC_ids) == len(predC_ids)

    gndD_ids = natsorted(next(os.walk(gndDiscPath))[2])
    predD_ids = natsorted(next(os.walk(predDiscPath))[2])

    assert len(gndD_ids) == len(predD_ids)

    print("Length: ", len(gndC_ids))

    GTc = []
    GTd = []

    PRc = []
    PRd = []

    for i in range(len(gndC_ids)):
        gndC = np.asarray(cv2.imread(os.path.join(gndCupPath, gndC_ids[i]), 0)) // 255
        preC = np.asarray(cv2.imread(os.path.join(predCupPath, predC_ids[i]), 0)) / 255.0

        gndD = np.asarray(cv2.imread(os.path.join(gndDiscPath, gndD_ids[i]), 0)) // 255
        preD = np.asarray(cv2.imread(os.path.join(predDiscPath, predD_ids[i]), 0)) / 255.0

        # To save memory resize and compute score
        gndC = cv2.resize(gndC, (100, 100), cv2.INTER_LINEAR)
        preC = cv2.resize(preC, (100, 100), cv2.INTER_LINEAR)

        gndD = cv2.resize(gndD, (100, 100), cv2.INTER_LINEAR)
        preD = cv2.resize(preD, (100, 100), cv2.INTER_LINEAR)

        GTc.extend(gndC.flatten())
        GTd.extend(gndD.flatten())

        PRc.extend(preC.flatten())
        PRd.extend(preD.flatten())

    fprC, tprC, tC = roc_curve(GTc, PRc)
    fprD, tprD, tD = roc_curve(GTd, PRd)

    return fprC, tprC, fprD, tprD


def Train_Val_Scores():
    s = "OC"
    t = s + "_out_N"

    dataset = "REFUGE"

    gndPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    TNt, FPt, FNt, TPt = compute_scores(gndPath, predPath)
    print("\n---Train Set Scores---")
    get_scores(TNt, FPt, FNt, TPt)

    gndPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predPath = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t

    TNv, FPv, FNv, TPv = compute_scores(gndPath, predPath)
    print("\n---Val Set Scores---")
    get_scores(TNv, FPv, FNv, TPv)

    print("\n---Overall Set Scores---")
    get_scores(TNt + TNv, FPt + FPv, FNt + FNv, TPt + TPv)


def Cross_Scores():
    gndPath = "D:/IISc/Python/MrRCNN-Disc_Cup/Messidor/OD"
    predPath = "D:/IISc/Python/MrRCNN-Disc_Cup/Messidor/OD_out_N"

    TN, FP, FN, TP = compute_scores(gndPath, predPath)
    print(TN, FP, FN, TP)
    print("\n---Overall Set Scores---")
    get_scores(TN, FP, FN, TP)


def plot_auc():
    dataset = "REFUGE"

    u = "_out_U"

    plt.rcParams['figure.dpi'] = 600

    if u == "_out_NU":
        plt.title("MrRCNN - " + dataset)
    else:
        plt.title("MRCNN - " + dataset)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    s = "OC"
    t = s + u

    gndCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predCupPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    gndCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predCupPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t

    s = "OD"
    t = s + u

    gndDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + s
    predDiscPathT = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/train/" + t

    gndDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + s
    predDiscPathV = "D:/IISc/Python/MrRCNN-Disc_Cup/" + dataset + "/val/" + t

    L = []

    fprC, tprC, fprD, tprD = get_roc_plots(gndCupPathT, predCupPathT, gndDiscPathT, predDiscPathT)
    plt.plot(fprC, tprC, c='b', ls='-')
    L.append(Line2D([0], [0], c='b', lw=2, mec='b', ls='-', label="Cup Train AUC: " + str(round(auc(fprC, tprC), 4))))

    plt.plot(fprD, tprD, c='c', ls='-')
    L.append(Line2D([0], [0], c='c', lw=2, mec='c', ls='-', label="Disc Train AUC: " + str(round(auc(fprD, tprD), 4))))

    fprC, tprC, fprD, tprD = get_roc_plots(gndCupPathV, predCupPathV, gndDiscPathV, predDiscPathV)
    plt.plot(fprC, tprC, c='orange', ls='-')
    L.append(Line2D([0], [0], c='orange', lw=2, ls='-', mec='orange',
                    label="Cup Val AUC: " + str(round(auc(fprC, tprC), 4))))

    plt.plot(fprD, tprD, c='m', ls='-')
    L.append(Line2D([0], [0], c='m', lw=2, ls='-', mec='m', label="Disc Val AUC: " + str(round(auc(fprD, tprD), 4))))

    plt.legend(handles=L, loc="lower right")
    plt.show()
    plt.close()


if __name__ == "__main__":

    ### NOTE: Make sure to link to the right directory path for predicted images and ground truth images
    gndPath = "./results/gn/"
    predPath = "./results/pred/"

    TN, FP, FN, TP = compute_scores(gndPath, predPath)
    print("\n---Performance Scores---")
    get_scores(TN, FP, FN, TP)
