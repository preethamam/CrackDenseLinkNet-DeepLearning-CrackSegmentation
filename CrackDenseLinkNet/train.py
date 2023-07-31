import pickle
import warnings
import logging
import os
import cv2
import gc
import argparse
import setproctitle
import shutil
import random
import time


import numpy as np
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics
import torch_utils as utils
from torch_utils import *

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def add_args(parser):

    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument('--model', type=str, default='Linknet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--backbone', type=str, default='densenet169',
                        help='employ with backbone (default: xception)')

    parser.add_argument('--batch_size', type=int, default=7, metavar='N',
                        help='input batch size for training (default: 32)')

    parser.add_argument('--epochs', type=int, default=200, metavar='EP',
                        help='how many epochs will be trained locally')
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--device_num', type=str, default="2",
                        help='gpu_server_num')

    parser.add_argument('--data_dir', type=str, default="/Liu+Xincong+CrackSegNet+CDLN",
                        help='Directory of Dataset which contains Testing and Ground Truth sub folders')

    args = parser.parse_args()
    return args    



def create_model(args, classes, activation, aux_params=None):

    """Model Initializer
    Args:
        args: Input arguments parsed through parser
        classes: Number of output classes
        activation: Activation Method
    Return:
        Initalized Segmentaion Model
    """

    model_name = args.model
    BACKBONE = args.backbone            
    
    if model_name == 'aaaat':
        logging.info('Creating Model: Linknet;Backbone: {}'.format(BACKBONE))
       
    else:
        if model_name == 'Unet':
            logging.info('Creating Model: Unet; Backbone: {}'.format(BACKBONE))
            model = smp.Unet(BACKBONE, classes=classes, activation=activation)
        elif model_name == 'UnetPlusPlus':
            logging.info('Creating Model: UnetPlusPlus; Backbone: {}'.format(BACKBONE))
            model = smp.UnetPlusPlus(BACKBONE, classes=classes, activation=activation)
        elif model_name == 'Linknet':
            logging.info('Creating Model: Linknet; Backbone: {}'.format(BACKBONE))
            model = smp.Linknet(BACKBONE, classes=classes, activation=activation,encoder_weights='imagenet') #,encoder_depth=4)
        elif model_name == 'FPN':
            logging.info('Creating Model: FPN; Backbone: {}'.format(BACKBONE))
            model = smp.FPN(BACKBONE, classes=classes, activation=activation)
        elif model_name == 'PSPNet':
            logging.info('Creating Model: PSPNet; Backbone: {}'.format(BACKBONE))
            model = smp.PSPNet(BACKBONE, classes=classes, activation=activation)
        elif model_name == 'PAN':
            logging.info('Creating Model: PAN; Backbone: {}'.format(BACKBONE))
            model = smp.PSPNet(BACKBONE, classes=classes, activation=activation)
        elif model_name == 'DeepLabV3':
            logging.info('Creating Model: DeepLabV3; Backbone: {}'.format(BACKBONE))
            model = smp.DeepLabV3(BACKBONE, classes=classes, activation=activation)
        elif model_name == 'DeepLabV3Plus':
            logging.info('Creating Model: DeepLabV3Plus; Backbone: {}'.format(BACKBONE))
            model = smp.DeepLabV3Plus(BACKBONE, classes=classes, activation=activation)
        
    return model


if __name__ == "__main__":


    # Start counter 
    start = time.time()

    parser = argparse.ArgumentParser()
    args = add_args(parser)

    DevNum = args.device_num
    
    # Set GPU 
    os.environ["CUDA_VISIBLE_DEVICES"]=DevNum # 0, 1, 2 and 3

    # Logs path
    LOGS_PATH = './logs/2023-TverskyLoss-SMP-smooth-1.0-alpha-0.8-beta-0.8-gamma-5.0'

    # Loss function
    # All CDLN learning rate = 0.0001
    ALPHA = 0.1 # Best ALPHA = 0.9
    GAMMA = 1 # Best GAMMA = 5
    BETA =  1.0 # mybeta = 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99
    # loss = utils.DiceFocalLoss(ALPHA=ALPHA, GAMMA=GAMMA, BETA=BETA)
    # loss = utils.DiceFocalLossSMP(BETA=BETA, smooth=1.0, alpha=0.9, gamma=5.0, log_loss=False)
    # loss = smp.utils.losses.BCEWithLogitsLoss(pos_weight=torch.tensor(15.1383))
    # loss = smp.utils.losses.JaccardLoss()
    loss = smp.losses.TverskyLoss('binary', smooth=1.0, alpha=0.8, beta=0.8, gamma=5.0)
    # loss = smp.losses.DiceLoss('binary', smooth=1.0, log_loss=False)
    # loss = smp.losses.FocalLoss('binary', alpha=0.9, gamma=5.0)
    loss_function_name = 'TverskyLoss'
    loss.__name__ = 'TverskyLoss'

	# Select the GPU on which training to be done -> select from [0, 3] for kraken as it supports 4 GPUs.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.simplefilter(action='ignore', category=FutureWarning)

     # customize the log format
    logging.basicConfig(level=logging.INFO,
                        format=str(
                            DEVICE) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')


    logging.info('Given arguments {0}'.format(args))

    # customize the process name
    str_process_name = "{model}-{backbone}-b{batch_size}e{epochs}-train".format(
                        model=args.model,
                        backbone=args.backbone,
                        batch_size=args.batch_size,
                        epochs=args.epochs)
                        
    setproctitle.setproctitle(str_process_name)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)    

    # Link dataset directory
    DATA_DIR = args.data_dir


    # check if data path exists
    if not os.path.exists(DATA_DIR):
        logging.info('Failed to load data...')
        exit(-1)
    
    DATA_DIR = "/Liu+Xincong+CrackSegNet+CDLN"

    # Training data
    x_train_dir = os.path.join(DATA_DIR, 'TrainingCracks')
    y_train_dir = os.path.join(DATA_DIR, 'TrainingCracksGroundtruth')

    # Validation data
    VALIND_DIR = "/Liu+Xincong+CrackSegNet+CDLN/ValidationCracksIndividual"
    x_valid_dir_Xincong = os.path.join(VALIND_DIR, 'Xincong','ValidCracks')
    y_valid_dir_Xincong = os.path.join(VALIND_DIR, 'Xincong', 'ValidCracksGroundtruth')

    x_valid_dir_DS3 = os.path.join(VALIND_DIR, 'DS3', 'ValidCracks')
    y_valid_dir_DS3 = os.path.join(VALIND_DIR, 'DS3','ValidCracksGroundtruth')

    x_valid_dir_CrackSegNet = os.path.join(VALIND_DIR, 'CrackSegNet', 'ValidCracks')
    y_valid_dir_CrackSegNet = os.path.join(VALIND_DIR, 'CrackSegNet','ValidCracksGroundtruth')

    x_valid_dir_Liu = os.path.join(VALIND_DIR, 'Liu', 'ValidCracks')
    y_valid_dir_Liu = os.path.join(VALIND_DIR, 'Liu','ValidCracksGroundtruth')

    VALID_DIR = "/Liu+Xincong+CrackSegNet+CDLN"
    x_valid_dir = os.path.join(VALID_DIR,  'ValidationCracks')
    y_valid_dir = os.path.join(VALID_DIR,  'ValidationCracksGroundtruth')

	
	# Change string to get different models -> {resnet18, resnet34 etc)
	# Check Segmentation Models Github documentation for reference
    BACKBONE = args.backbone
	
	# We have only one class - crack - binary segmentation problem [0 - no crack, 1 - crack]
    CLASSES = ['crack']

	# Low learning rate -> Can use LRPlateau in Callbacks for LR degrada

	# Preprocess backbone -> check Github doc for better info
    preprocess_input = get_preprocessing_fn(BACKBONE, pretrained='imagenet')

    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'		# in this case its sigmoi


	# Dataset for train images
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        classes=CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Dataset for validation images
    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    
    valid_dataset_Liu = Dataset(
        x_valid_dir_Liu,
        y_valid_dir_Liu,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dataset_Xincong = Dataset(
        x_valid_dir_Xincong,
        y_valid_dir_Xincong,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dataset_CrackSegNet = Dataset(
        x_valid_dir_CrackSegNet,
        y_valid_dir_CrackSegNet,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )
    valid_dataset_DS3 = Dataset(
        x_valid_dir_DS3,
        y_valid_dir_DS3,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    # Use DataLoader defined in utils.py
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, num_workers=12)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1, num_workers=12)
    valid_dataloader_Liu = DataLoader(valid_dataset_Liu, batch_size=1, shuffle=False, num_workers=4)
    valid_dataloader_Xincong = DataLoader(valid_dataset_Xincong, batch_size=1, shuffle=False, num_workers=4)    
    valid_dataloader_CrackSegNet = DataLoader(valid_dataset_CrackSegNet, batch_size=1, shuffle=False, num_workers=4)
    valid_dataloader_DS3 = DataLoader(valid_dataset_DS3, batch_size=1, shuffle=False, num_workers=4)

    # create model
    model = create_model(args, n_classes, activation)

    model = nn.DataParallel(model, device_ids=[0])
    model.train()

    # define optimizer     
    # lr = 0.00001 #0.00001, 0.0001, 0.0005, 0.001, and 0.005 
    lr = args.lr
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=lr)]) #SOTA

	# Choose metrics
    metrics = [smp.utils.metrics.IoU(threshold=0.5), 
    smp.utils.metrics.Fscore(threshold=0.5), 
    smp.utils.metrics.Precision(threshold=0.5),
    smp.utils.metrics.Recall(threshold=0.5),
    smp.utils.metrics.Accuracy(threshold=0.5)]

    # create epoch runners 
    # It is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=DEVICE,
        verbose=True,
    )
    
    ### Train Model
    save_dir = os.path.join(LOGS_PATH, args.model, args.backbone)

    ### Set patience value to stop after n iterations if validation fscore does not imporve
    # early_stopping = EarlyStopping(patience=20, verbose=True, path=os.path.join(save_dir, 'best_model_iou.pth'))

    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            logging.info("Creation of the directory %s failed" % save_dir)
        else:
            logging.info("Successfully created the directory %s " % save_dir)

    # LossName = "DiceFocalLoss"
    LossName = loss_function_name
    train_record = record(LossName)
    valid_record = record(LossName)
    valid_record_Liu = record(LossName)
    valid_record_Xincong = record(LossName)
    valid_record_DS3 = record(LossName)
    valid_record_CrackSegNet = record(LossName)
    count = 0
    val_score_previous = -10

    for i in range(0, args.epochs):

        logging.info('\nEpoch: {}'.format(i))

        ### Run the lop and get logs
        # /home/preethamam/miniconda3/envs/vision/lib/python3.9/site-packages/segmentation_models_pytorch/utils/train.py
        train_logs = train_epoch.run(train_dataloader)

        #validation
        valid_logs = valid_epoch.run(valid_dataloader)
        valid_logs_Liu = valid_epoch.run(valid_dataloader_Liu)
        valid_logs_CrackSegNet = valid_epoch.run(valid_dataloader_CrackSegNet)
        valid_logs_Xincong = valid_epoch.run(valid_dataloader_Xincong)
        valid_logs_DS3 = valid_epoch.run(valid_dataloader_DS3)

        #log detailed training information
        print(train_logs.keys())
        train_record.log(train_logs,i)
        valid_record.log(valid_logs,i)
        valid_record_Liu.log(valid_logs_Liu,i)
        valid_record_CrackSegNet.log(valid_logs_CrackSegNet, i)
        valid_record_Xincong.log(valid_logs_Xincong, i)
        valid_record_DS3.log(valid_logs_DS3, i)
        
        ### Selecting metric to track for early stopping
        val_score = valid_logs['fscore']
        flag = False # early_stopping(val_score, model, i, optimizer)
        if flag:
            train_record.write(save_dir, 'train')
            valid_record.write(save_dir, 'valid')
            valid_record_Liu.write(save_dir,'Liu')
            valid_record_Xincong.write(save_dir, 'Xincong')
            valid_record_DS3.write(save_dir, 'DS3')
            valid_record_CrackSegNet.write(save_dir, 'CrackSegNet')
            logging.info('Model Successfully Trained!')
        ### Record metrics
        ### Uncomment below three lines if you want it to early stop based on patience value
        # if early_stopping.early_stop:
        #     logging.info("Early stopping")
        #     break

        ### Following piece of code was a part of experimentation while finetuning. It drops the learning rate after couple of iterations
        ### SOTA results were achieved without this type of special tweaking hence keep it commented unless you want to try playing around with this approach
        # if i == 25:
        #     optimizer.param_groups[0]['lr'] = 1e-5
        #     logging.info('Decrease decoder learning rate to 1e-5!')
        
        if val_score > val_score_previous:
            torch.save({'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'f_score': val_score}, LOGS_PATH+'/best_model_iou.pth')
            val_score_previous = val_score
                      


    save_dir = os.path.join(LOGS_PATH, '3Data' , args.model, args.backbone, str(args.batch_size), LossName, str(lr), 'NoneWeights','metrics')
    print(save_dir)

    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError:
            logging.info("Creation of the directory %s failed" % save_dir)
        else:
            logging.info("Successfully created the directory %s " % save_dir)


    train_record.write(save_dir, 'train')
    valid_record.write(save_dir, 'valid')
    valid_record_Liu.write(save_dir,'Liu')
    valid_record_Xincong.write(save_dir, 'Xincong')
    valid_record_DS3.write(save_dir, 'DS3')
    valid_record_CrackSegNet.write(save_dir, 'CrackSegNet')
    logging.info('Model Successfully Trained!')

    # Stop counter
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Elapsed time: ' "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
