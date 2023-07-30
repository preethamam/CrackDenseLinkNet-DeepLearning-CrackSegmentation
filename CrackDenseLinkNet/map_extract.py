import argparse
import os
import cv2
import sys
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision

import matplotlib
import matplotlib.pyplot as plt
import albumentations as A

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

IMG_DIMS = 512
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_children(model: torch.nn.Module):
    """Generate Feature Maps
    Args:
        model: Segmentation Model intitalized from smp library
    Return:
        flatt_children: List of deeply neseted flattened children
    """

    flatt_children = []

    children = list(model.children())

    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def save_plot(plt, fig, save_dir):
    """Plot Saver
    Args:
        plt: matplotlib pyplot
        fig: figure of the plot
        save_dir: path to save the generated filter map plot
    Return:
        None
    """ 
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)    
    plt.savefig(save_dir + '.png', bbox_inches='tight', pad_inches = 0)

def vistensor(tensor, save_dir, ch=0, allkernels=False, nrow=8, padding=1): 
    """Visuzlization Tensor
    Args:
        tensor: image/kernel tensor
        save_dir: path to save the generated filter map plot
        ch: visualization channel 
        allkernels: visualization all tensores
        nrow: number of rows
        padding: pads to add
    Return:
        None
    """ 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows))
    fig = plt.imshow(grid.numpy().transpose((1, 2, 0)), cmap=matplotlib.cm.jet, aspect='auto')
    save_plot(plt, fig, save_dir)

def generate_filter_maps(conv_layers, weights, filter_dir, layer_num):
    """Filter Maps Generator
    Args:
        conv_layers: List of all convolutional layers of the model
        weights: List of weights corresponding to the conv layers
        layer_num: layer_num to generate the specific filter map
    Return:
        None
    """ 

    if not layer_num < len(weights):
        print('Given Layer num exceeds the maximum CNN layer present in the architecture. Extracting the map of the last layer:')
        layer_num = len(weights)-1
    save_dir = os.path.join(filter_dir, str(conv_layers[layer_num]))
    kernel = weights[layer_num].data.clone()
    vistensor(kernel, save_dir, ch=0, allkernels=False)

def generate_feature_maps(model_blocks, tensor, save_path, layer_num):
    """Generate Feature Maps
    Args:
        model_blocks: List of first layer encoder + decoder blocks
        tensor: processed and permuted image tensor
        save_path: location to save feature map
        layer_num: layer number to extract feature map from
    Return:
        None
    """

    act = tensor
    for i,layer in enumerate(model_blocks):
        act = layer(act)
        if (i >= layer_num):
            break 
    act = act.detach().cpu().squeeze()
    for idx in range(act.size(0)):
        fig = plt.imshow(act[idx], cmap=matplotlib.cm.jet, aspect='auto')

    save_plot(plt, fig, save_path)

def augmentation():
    """Add paddings to make image shape divisible by 32"""

    test_transform = [
        A.PadIfNeeded(IMG_DIMS, IMG_DIMS)
    ]
    return A.Compose(test_transform)

def preprocessing(preprocessing_fn):
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

def process_image(image, preprocess_function):
    """Image Processing

    Args:
        Input Image
    Return:
        Processed Image
    """

    ### Rescale the images
    scale_h = img_rows / image.shape[0]
    scale_w = img_cols / image.shape[1]
    image = cv2.resize(image, dsize=(img_rows, img_cols), fx=scale_h, fy=scale_w)

    ### Image Augmentation
    sample = augmentation()(image = image)
    image = sample['image']

    ### Image Preprocessing
    sample = preprocessing(preprocess_function)(image=image)
    image = sample['image']
    image = torch.from_numpy(image).permute(2,0,1).float()

    return image


def add_args(parser):
    """Argumet Parser
    Args:
        parser : argparse.ArgumentParser
    Return:
        args: parsed arguments
    """

    # Training settings

    # Add parser argument for "--model"
    parser.add_argument('--model', type=str, default='Linknet', metavar='N',
                        help='neural network used in training')

    # Add parser argument for "--backbone"
    parser.add_argument('--backbone', type=str, default='densenet169',
                        help='employ with backbone (default: xception)')
    
    parser.add_argument('--device_num', type=int, default=0,
                        help='gpu_server_num')

    # Add parser argument for "--data_dir"
    parser.add_argument('--data_dir', type=str, default='./data//TestingCracks/',
                        help='Directory of Dataset which contains Testing Images')

    # Add parser argument for "--image_name"
    parser.add_argument('--image_name', type=str, default='11215-11.jpg',
                        help='Name of testing image')

    args = parser.parse_args()
    return args    
    
    args = parser.parse_args()
    return args    

def create_model(args, classes, activation):
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
    
    if model_name == 'Unet':
        model = smp.Unet(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'Linknet':
        model = smp.Linknet(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'FPN':
        model = smp.FPN(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'PSPNet':
        model = smp.PSPNet(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'PAN':
        model = smp.PSPNet(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'DeepLabV3':
        model = smp.DeepLabV3(BACKBONE, classes=classes, activation=activation)
    elif model_name == 'DeepLabV3Plus':
        model = smp.DeepLabV3(BACKBONE, classes=classes, activation=activation)
    
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = add_args(parser)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Creating directories to save filter maps and feature maps
    filter_dir = './filter_maps/'
    feature_dir = './feature_maps/' + args.image_name.split('.')[0]
    dirs = [filter_dir, feature_dir]

    for path in dirs:
        if os.path.exists(path):
            shutil.rmtree(path)
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)

    ### Get Preprocessing Function
    preprocess_function = get_preprocessing_fn(args.backbone, pretrained='imagenet')

    ### Set Image Dimensions
    img_rows, img_cols = IMG_DIMS, IMG_DIMS

    # Load and process Image
    image_path = os.path.join(args.data_dir, args.image_name)
    image = cv2.imread(image_path)
    image = process_image(image, preprocess_function)

    # Parameters for model
    n_classes = 1
    activation = 'sigmoid'

    # Load saved checkpoint
    PATH = os.path.join('./logs/', args.model, args.backbone, 'best_model_iou.pth')


    best_model = create_model(args, n_classes, activation)
    checkpoint = torch.load(PATH, map_location=DEVICE)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    # Populate get_children function to flatten the model architecture
    model_children = get_children(best_model)

    print('Total Number of Layers:', len(model_children))

    conv_layers = []
    model_weights = []

    # Fetch all convolutional layers
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])

    print('Total Conv Layers: ', len(conv_layers))

    # Generating and saving filter maps
    for i in range(len(conv_layers)):
        if i%25 == 0:
            generate_filter_maps(conv_layers, model_weights, filter_dir, i)


    model_blocks = []

    # Append all the features of encoder to model_blocks
    model_blocks.extend(best_model.encoder.features)

    # Append all the blocks of decoder to model_blocks
    model_blocks.extend(best_model.decoder.blocks)
    
    print('Total blocks: ',  len(model_blocks))

    # Set model to eval
    best_model.to(DEVICE)
    best_model.eval()

    # Align the image dimensions
    x_tensor = image.to(DEVICE).unsqueeze(0)

    for i in range(len(model_blocks)):
        save_path = os.path.join(feature_dir, 'layer_' + str(i) +'.png')
        generate_feature_maps(model_blocks, x_tensor, save_path, i)