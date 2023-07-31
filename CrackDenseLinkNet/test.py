import argparse
import pickle
import warnings
import os
import cv2
import gc
import sys
import numpy
import torch
import torch.nn as nn
import torchvision
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics
import torch_utils as utils
import setproctitle
import shutil
import time
from torch_utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torchsummaryX import summary
from performance_analysis import *

numpy.set_printoptions(threshold=sys.maxsize)


def add_args(parser):

    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings

    parser.add_argument(
        "--model",
        type=str,
        default="Linknet",
        metavar="N",
        help="neural network used in training",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="densenet169",
        help="employ with backbone (default: xception)",
    )

    parser.add_argument("--device_num", type=int, default=0, help="gpu_server_num")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Liu+Xincong+CrackSegNet+CDLN/Testing/",
        help="Directory of Dataset which contains Testing and Ground Truth sub folders",
    )

    args = parser.parse_args()
    return args


def create_model(args, classes, activation):
    model_name = args.model
    BACKBONE = args.backbone

    if model_name == "Unet":
        logging.info("Creating Model: Unet; Backbone: {}".format(BACKBONE))
        model = smp.Unet(BACKBONE, classes=classes, activation=activation)
    elif model_name == "UnetPlusPlus":
        logging.info("Creating Model: UnetPlusPlus; Backbone: {}".format(BACKBONE))
        model = smp.UnetPlusPlus(BACKBONE, classes=classes, activation=activation)
    elif model_name == "Linknet":
        logging.info("Creating Model: Linknet; Backbone: {}".format(BACKBONE))
        model = smp.Linknet(
            BACKBONE, classes=classes, encoder_weights="imagenet", activation=activation
        )
    elif model_name == "FPN":
        logging.info("Creating Model: FPN; Backbone: {}".format(BACKBONE))
        model = smp.FPN(BACKBONE, classes=classes, activation=activation)
    elif model_name == "PSPNet":
        logging.info("Creating Model: PSPNet; Backbone: {}".format(BACKBONE))
        model = smp.PSPNet(BACKBONE, classes=classes, activation=activation)
    elif model_name == "PAN":
        logging.info("Creating Model: PAN; Backbone: {}".format(BACKBONE))
        model = smp.PSPNet(BACKBONE, classes=classes, activation=activation)
    elif model_name == "DeepLabV3":
        logging.info("Creating Model: DeepLabV3; Backbone: {}".format(BACKBONE))
        model = smp.DeepLabV3(BACKBONE, classes=classes, activation=activation)
    elif model_name == "DeepLabV3Plus":
        logging.info("Creating Model: DeepLabV3Plus; Backbone: {}".format(BACKBONE))
        model = smp.DeepLabV3(BACKBONE, classes=classes, activation=activation)

    return model


if __name__ == "__main__":

    ### Parse arguments
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_num)  # 0, 1, 2 and 3

    ### Set the GPU on which training to be done
    DEVICE = torch.device(
        "cuda:" + str(args.device_num) if torch.cuda.is_available() else "cpu"
    )

    # DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(DEVICE)

    ### Clear cache
    torch.cuda.empty_cache()
    gc.collect()

    ### Ignore warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)

    print(args)

    ### customize the process name
    str_process_name = args.model + "-" + args.backbone + "-test"
    setproctitle.setproctitle(str_process_name)

    ### Dataset directory
    DATA_DIR = args.data_dir

    ### check if data path exists
    if not os.path.exists(DATA_DIR):
        print("Failed to load data...")
        exit(-1)

    ### Testing data
    x_test_dir = os.path.join(DATA_DIR, "CDLN", "TestingCracks")
    y_test_dir = os.path.join(DATA_DIR, "CDLN", "TestingCracksGroundtruth")

    ### We have only one class - crack - binary segmentation problem [0 - no crack, 1 - crack]
    CLASSES = ["crack"]

    ### Choose metrics
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.Fscore(threshold=0.5),
    ]

    ### Load saved checkpoint
    PATH = os.path.join("./" "best_model_iou.pth")

    ### Define classes and activation function
    n_classes = (
        1 if len(CLASSES) == 1 else (len(CLASSES) + 1)
    )  # case for binary and multiclass segmentation
    activation = "sigmoid" if n_classes == 1 else "softmax"  # in this case its sigmoid

    preprocess_input = get_preprocessing_fn(args.backbone, pretrained="imagenet")

    ### Initialize and load best model for evaluation on GPU device
    best_model = create_model(args, n_classes, activation)

    # best_model = nn.DataParallel(best_model, device_ids=[0])
    checkpoint = torch.load(PATH, map_location=DEVICE)
    best_model.cuda()
    best_model.load_state_dict(checkpoint["model_state_dict"])
    best_model.eval()

    # Dataset for test images
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        classes=CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input),
    )

    test_dataloader = DataLoader(test_dataset)

    ids = np.arange(len(test_dataset))

    ### Create directories for saving imnage and ground truth
    dirs = ["./pred/tail/gnd/", "./pred/tail/seg/"]
    for path in dirs:
        if os.path.exists(path):
            shutil.rmtree(path)

        try:
            os.makedirs(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

    ### Set starter and ender for measuring inference time
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    total_time = 0
    softmax_dict = {}

    ### Run inference loop
    a = []
    for i in tqdm(ids):
        image, gt_mask, im_shape = test_dataset[i]
        x_tensor = image.to(DEVICE).unsqueeze(0)
        gt_mask = gt_mask.permute(1, 2, 0).cpu().numpy()

        start_time = time.time()
        pr_mask = best_model.predict(x_tensor)
        # pr_mask = best_model.forward(x_tensor)
        predict_time = time.time() - start_time
        total_time += predict_time

        ### WAIT FOR GPU SYNC
        torch.cuda.synchronize()

        ### Align dimensions
        pr_mask = pr_mask.permute(0, 2, 3, 1).detach().cpu().numpy().round()
        # pr_mask = (pr_mask.permute(0,2,3,1).cpu().numpy().round())

        ### Save predicted image and ground truth in respective directories
        id = test_dataset.train_ids[i].replace(".jpg", ".png")

        pr_mask = pr_mask[..., 0].squeeze() * 255
        gt_mask = gt_mask[..., 0].squeeze() * 255

        ## Save predicted image and ground truth in respective directories
        pred_final = cv2.resize(pr_mask, (im_shape[1], im_shape[0]), cv2.INTER_LINEAR)
        gt_final = cv2.resize(gt_mask, (im_shape[1], im_shape[0]), cv2.INTER_LINEAR)

        # Save images
        cv2.imwrite(dirs[0] + id, gt_final)
        cv2.imwrite(dirs[1] + id, pred_final)


### Measure elapsed time
elapsed_time = elapsed_time / len(test_dataset)


print("Average time: " + str(total_time / len(test_dataset)))
print("============>>>> Finish predict ... <<<<============")
