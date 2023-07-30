# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 11:16:13 2019

@author: vmani
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import shutil

from math import sqrt
from numba import jit
from natsort import natsorted
from scipy.stats import pearsonr
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc


def get_scores(TN, FP, FN, TP):
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    error_rate = (FP + FN) / (TP + FN + TN + FP)
    accuracy = 1 - error_rate
    precision = TP / (TP + FP)
    dice_coeff = (2 * TP) / (2 * TP + FP + FN)
    iou = dice_coeff / (2 - dice_coeff)
    f_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, specificity, iou, precision, recall, f_score


@jit(nopython=True)
def confusion_matrix(pre, gnd, TN, FP, FN, TP):
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
                
    return TN, FP, FN, TP


def compute_scores(gndPath, predPath, threshold=1):
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    gnd_ids = natsorted(next(os.walk(gndPath))[2])
    pred_ids = natsorted(next(os.walk(predPath))[2])

    assert len(gnd_ids) == len(pred_ids)

    for i in range(len(gnd_ids)):
        
        gnd = cv2.imread(os.path.join(gndPath, gnd_ids[i]), cv2.IMREAD_GRAYSCALE)
        pre = cv2.imread(os.path.join(predPath, pred_ids[i]), cv2.IMREAD_GRAYSCALE)

        pre = cv2.resize(pre, (gnd.shape[1], gnd.shape[0]), cv2.INTER_AREA)
        pre[pre <= threshold] = 0
        pre[pre > threshold] = 255
        gnd[gnd <= threshold] = 0
        gnd[gnd > threshold] = 255

        TN, FP, FN, TP = confusion_matrix(pre, gnd, TN, FP, FN, TP)

    return TN, FP, FN, TP


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Unet', metavar='N',
                        help='neural network used in training')
    args = parser.parse_args()

    ### NOTE: Make sure to link to the right directory path for predicted images and ground truth images
    #gndPath = "./pred/tail/gnd/"

    predPath = "./pred/tail/seg/"
    gndPath =  os.path.join("/media/preethamam/Utilities-SSD-1/Xtreme_Programming/Z_Data/DLCrack/Liu+Xincong+DS3+CrackSegNet/Testing",args.data,"TestingCracksGroundtruth")
    print(gndPath) 
    results = []
    for threshold in range(253,254):
        TN, FP, FN, TP = compute_scores(gndPath, predPath, threshold)
           
        # print("\n---"+args.data+" Performance Scores---")
        # get_scores(TN, FP, FN, TP)

        get_scores(TN, FP, FN, TP)
        #results.append([threshold, recall,precision,fscore])

    # results = np.array(results)
    # np.savetxt(args.data + ".csv", 
    #        results,
    #        delimiter =", ", 
    #        fmt ='% s')
