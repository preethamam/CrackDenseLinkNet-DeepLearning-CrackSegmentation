import os
import cv2
from tqdm import tqdm
import logging
import hdf5storage

from collections import OrderedDict

import numpy as np
import albumentations as A
from natsort import natsorted
from skimage import exposure
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset as BaseDataset
import torch.nn as nn
import torch.nn.functional as F
import re
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics

#img_dims = 512
img_dims_width = 512
img_dims_height = 512

class SensitivitySpecificityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def __name__(self):
        return 'SensitivitySpecificityLoss'

    def forward(self, inputs, targets, alpha=0.8, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Sensitivity
        Sensitivity = ((torch.square(inputs-targets)*targets).sum())/(targets.sum() + smooth)

        # Specificity
        Specificity = ((torch.square(inputs-targets)*(1-targets)).sum())/(((1-targets).sum()) + smooth)

        loss = alpha * Sensitivity + (1 - alpha) * Specificity

        return loss

        

class TverskyFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def __name__(self):
        return 'TverskyFocalLoss'

    def forward(self,inputs,targets,alpha=0.7, beta=0.3, gamma=(4.0/3.0),smooth=1):
         
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #BCE
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        
        #focal loss
        #focal_loss = self.alpha * (1-BCE_EXP)**self.gamma*BCE
        
        #TverskyFocalLoss
        TP = (inputs * targets).sum()
        FN = ((1-inputs) * targets).sum()
        FP = ((1-targets) * inputs).sum()

        #tverskyFocalLoss
        tverskyIndex = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
        tverskyFocalLoss = 1 - tverskyIndex ** gamma
        return tverskyFocalLoss
    

class TverskyFocalFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def __name__(self):
        return 'TverskyFocalFocalLoss'

    def forward(self, inputs, targets, alpha_tversky=0.3, beta_tversky=0.7, gamma_tversky=(4.0/3.0), 
                alpha=0.01, gamma=5.0, smooth = 1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #BCE
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)

        #focal loss
        focal_loss = alpha * (1-BCE_EXP)**gamma*BCE

        #TverskyFocalLoss
        TP = (inputs * targets).sum()
        FN = ((1-inputs) * targets).sum()
        FP = ((1-targets) * inputs).sum()
        tverskyIndex = (TP + smooth) / (TP + alpha_tversky * FN + beta_tversky * FP + smooth)
        tverskyFocalLoss = 1 - tverskyIndex ** gamma_tversky
        return tverskyFocalLoss + focal_loss

class DiceFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, ALPHA=0.0, GAMMA=5.0, BETA=0.5):
        super().__init__()
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.beta  = BETA
    @property
    def __name__(self):
        return 'DiceFocalLoss'      

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        #Dice Loss
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  

        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)

        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
        
        return (self.beta * focal_loss + (1 - self.beta) * dice_loss)

class DiceFocalLossSMP(nn.Module):
    def __init__(self, weight=None, size_average=True, BETA=0.5, smooth=1.0, alpha=0.9, gamma=5.0, log_loss=False):
        super().__init__()
        self.dice = smp.losses.DiceLoss('binary', smooth=smooth, log_loss=log_loss)
        self.focal = smp.losses.FocalLoss('binary', alpha=alpha, gamma=gamma)
        self.beta  = BETA
    @property
    def __name__(self):
        return 'SMPDiceFocalLoss'      

    def forward(self, inputs, targets):        
        return (self.beta * self.focal(inputs, targets) + (1 - self.beta) * self.dice(inputs, targets))        

# classes for data loading and preprocessing
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['non-crack', 'crack']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
    
        print(images_dir)
        self.train_ids = natsorted(next(os.walk(images_dir))[2])
        self.mask_ids = natsorted(next(os.walk(masks_dir))[2])

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.train_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.img_rows = img_dims_width
        self.img_cols = img_dims_height

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])

        im_shape = image.shape
     
        #DEBUG; TO REMOVE
        # dirs = ['./pred/tail/org/', './pred/tail/equal/']
        # cv2.imwrite(dirs[0] + str(i) + ".png", image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # image[:, :, 2] = cv2.equalizeHist(image[:, :, 2])    
        # image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
                
        mask = cv2.imread(self.masks_fps[i], 0)


        # Resizing
        scale_h = self.img_rows / image.shape[0]
        scale_w = self.img_cols / image.shape[1]
        
        image = cv2.resize(image, dsize=(self.img_rows, self.img_cols), fx=scale_h, fy=scale_w)
        mask = cv2.resize(mask, dsize=(self.img_rows, self.img_cols), fx=scale_h, fy=scale_w)
        mask[mask > 0] = 1

        # Padding 0
        # delta_w = self.img_cols - image.shape[1]
        # delta_h = self.img_rows - image.shape[0]
        # print("delta_w ", delta_w,"delta_h ", delta_h)
        # print("shape_h ", image.shape[0], "shape_w ", image.shape[1])
        # top, bottom = delta_h//2, delta_h-(delta_h//2)
        # left, right = delta_w//2, delta_w-(delta_w//2)
        # print(top,bottom,left,right)
        # image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #     value=[0, 0, 0])
        # mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT,
        #     value=0)

        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
    
        image = torch.from_numpy(image).permute(2,0,1).float()
        mask = torch.from_numpy(mask).permute(2,0,1).float()
        return image, mask, im_shape

    def __len__(self):
        return len(self.train_ids)

class EarlyStopping:
    
    """Early stops the training if validation score doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model_iou.pth', trace_func=logging.info):
        """
        Args:
            patience (int): How long to wait after last time validation score improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation score improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model, epoch, optimizer):

        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, epoch, optimizer)
            return True
        elif score <= self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, epoch, optimizer)
            self.counter = 0
            return True

    def save_checkpoint(self, val_score, model, epoch, optimizer):
        '''Saves model when validation score increases.'''
        if self.verbose:
            self.trace_func(f'Validation score increased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f_score': val_score}, self.path+"_"+str(epoch))
        self.val_score_min = val_score


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=img_dims_height, min_width=img_dims_height, always_apply=True, border_mode=0),
        A.RandomCrop(height=img_dims_height, width=img_dims_width, always_apply=True),

        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)
#def get_training_augmentation():
#    train_transform = [
#
#        A.HorizontalFlip(p=0.5),
#
#        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.2, p=0.5, border_mode=0),
#
#        A.VerticalFlip(p=0.4),
#
#        #A.RandomCrop(height=250, width=250, p=0.2),
#       
#        A.PadIfNeeded(min_height=img_dims_height, min_width=img_dims_height, p=1),
#        A.IAAAdditiveGaussianNoise(p=0.3),
#        A.IAAPerspective(p=0.5),
#
#        A.OneOf(
#            [
#                A.CLAHE(p=1),
#                A.RandomBrightness(p=1),
#                A.RandomGamma(p=1),
#            ],
#            p=0.2,
#        ),
#
#        A.OneOf(
#             [
#                 A.IAASharpen(p=1),
#                 A.Blur(blur_limit=3, p=1),
#                 A.MotionBlur(blur_limit=3, p=1),
#             ],
#             p=0.2,
#        ),
#
#        A.OneOf(
#            [
#                A.RandomContrast(p=1),
#                A.HueSaturationValue(p=1),
#            ],
#            p=0.2,
#        ),
#        A.Lambda(mask=round_clip_0_1)
#    ]
#    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
    A.PadIfNeeded(img_dims_width,img_dims_height)        
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')
    plt.show()

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1, voxel_spacing=None, option=1):

    """
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxel_spacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    if option == 1:
        def condgradient(delta, spacing):
            return np.exp(-(delta/kappa)**2.)/float(spacing)

        # initialize output array
        out = np.array(img, dtype=np.float32, copy=True)

        # set default voxel spacing if not supplied
        if voxel_spacing is None:
            voxel_spacing = tuple([1.] * img.ndim)

        # initialize some internal variables
        deltas = [np.zeros_like(out) for _ in range(out.ndim)]

        for _ in range(niter):

            # calculate the diffs
            for i in range(out.ndim):
                slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
                deltas[i][slicer] = np.diff(out, axis=i)

            # update matrices
            matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxel_spacing)]

            # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
            # pixel. Don't as questions. just do it. trust me.
            for i in range(out.ndim):
                slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
                matrices[i][slicer] = np.diff(matrices[i], axis=i)

            # update the image
            out += gamma * (np.sum(matrices, axis=0))

        return out

    elif option == 2:
        def condgradient(delta, spacing):
            return 1./(1.+(delta/kappa)**2.)/float(spacing)

        # initialize output array
        out = np.array(img, dtype=np.float32, copy=True)

        # set default voxel spacing if not supplied
        if voxel_spacing is None:
            voxel_spacing = tuple([1.] * img.ndim)

        # initialize some internal variables
        deltas = [np.zeros_like(out) for _ in range(out.ndim)]

        for _ in range(niter):

            # calculate the diffs
            for i in range(out.ndim):
                slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
                deltas[i][slicer] = np.diff(out, axis=i)

            # update matrices
            matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxel_spacing)]

            # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
            # pixel. Don't as questions. just do it. trust me.
            for i in range(out.ndim):
                slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
                matrices[i][slicer] = np.diff(matrices[i], axis=i)

            # update the image
            out += gamma * (np.sum(matrices, axis=0))

        return out

    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5*((1.-(delta/kappa_s)**2.)**2.)/float(spacing)
            return np.where(np.abs(delta) <= kappa_s, top, 0)

        # initialize output array
        out = np.array(img, dtype=np.float32, copy=True)

        # set default voxel spacing if not supplied
        if voxel_spacing is None:
            voxel_spacing = tuple([1.] * img.ndim)

        # initialize some internal variables
        deltas = [np.zeros_like(out) for _ in range(out.ndim)]

        for _ in range(niter):

            # calculate the diffs
            for i in range(out.ndim):
                slicer = [slice(None, -1) if j == i else slice(None) for j in range(out.ndim)]
                deltas[i][slicer] = np.diff(out, axis=i)

            # update matrices
            matrices = [condgradient(delta, spacing) * delta for delta, spacing in zip(deltas, voxel_spacing)]

            # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
            # pixel. Don't as questions. just do it. trust me.
            for i in range(out.ndim):
                slicer = [slice(1, None) if j == i else slice(None) for j in range(out.ndim)]
                matrices[i][slicer] = np.diff(matrices[i], axis=i)

            # update the image
            out += gamma * (np.sum(matrices, axis=0))

        return out


def display_img(img_list):

    for i in range(len(img_list)):
        # print(i)
        cv2.imshow(str(i), img_list[i])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

import json
class record():

    def __init__(self, loss_name):
        self.loss_name = loss_name
        self.logs = {
            "recall":[],
            "loss":[],
            "iou_score":[],
            "precision":[],
            "fscore":[],
            "accuracy":[],
            "epochs":[]
        }

    def log(self, train_logs, i):
        #train
        # self.logs['loss'].append(train_logs[self.loss_name])
        self.logs["iou_score"].append(train_logs['iou_score'])
        self.logs["fscore"].append(train_logs['fscore'])
        self.logs['precision'].append(train_logs['precision'])
        self.logs['recall'].append(train_logs['recall'])
        self.logs['accuracy'].append(train_logs['accuracy'])
        self.logs['epochs'].append(i)


    def write(self,save_dir, name):
        ### Record the metrics and store it in a dict to generate a matfile which can be used for plots
        save_dir = os.path.join(save_dir,name+'.json')
        print(save_dir)
        with open(save_dir, 'w') as fp:
            json.dump(self.logs, fp)



