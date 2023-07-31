# CrackDenseLinkNet (CDLN)
## Title
<p align="justify"> CrackDenseLinkNet: A deep convolutional neural network for semantic segmentation of cracks on concrete surface images </p>

## Abstract
<p align="justify">
Cracks are the defects formed by cyclic loading, fatigue, shrinkage, creep, and so on. In addition, they represent the deterioration of the structures over some time. Therefore, it is essential to detect and classify them according to the condition grade at the early stages to prevent the collapse of structures. Deep learning-based semantic segmentation convolutional neural network (CNN) has millions of learnable parameters. However, depending on the complexity of the CNN, it takes hours to days to train the network fully. In this study, an encoder network DenseNet and modified LinkNet with five upsampling blocks were used as a decoder network. The proposed network is referred to as the ‘‘CrackDenseLinkNet’’ in this work. CrackDenseLinkNet has 19.15 million trainable parameters, although the input image size is 512 x 512 and has a deeper encoder. CrackDenseLinkNet and four other state-of-the-art (SOTA) methods were evaluated on three public and one private datasets. The proposed CNN, CrackDenseLinkNet, outperformed the best SOTA method, CrackSegNet, by 2.2% of F1-score on average across the four datasets. Lastly, a crack profile analysis demonstrated that the CrackDenseLinkNet has lesser variance in relative errors for the crack width, length, and area categories against the ground-truth data.
</p>

## CrackDenseLinkNet Python modules installation and usage
### Setting up the Anaconda Environment

Please make sure you have installed Anaconda and have Python version `> 3.8.5` before following the next steps

```shell
conda env create -f crackdenselinknet.yml
conda activate crackdenselinknet
```

This should install all the necessary packages required to run the below-documented scripts. However, by any chance, if still you get package errors then just install that package through pip or conda.


## CrackDenseLinkNet and other methods trained model/weights files


## CrackDenseLinkNet, FCN, DeepCrack, and CrackSegNet datasets combined for training, validation, and testing
| ![All data](https://github.com/preethamam/CrackDenseLinkNet-DeepLearning-CrackSegmentation/assets/28588878/3b2c953f-e499-434f-9bfb-0ea701ecd9a7) | 
|:--:| 
| *Datasets’ sample images: (a) FCN, (b) DeepCrack, (c) CrackSegNet, and (d) CrackDenseLinkNet.* |
## CrackDenseLinkNet dataset only

## Citation
CrackDenseLinkNet code and dataset are available to the public. If you use this code/dataset in your research, please use the following BibTeX entry to cite:
```bibtex
@article{manjunatha2023crackdenselinknet,
author = {Preetham Manjunatha and Sami F Masri and Aiichiro Nakano and Landon Carter Wellford},
title ={{CrackDenseLinkNet}: a deep convolutional neural network for semantic segmentation of cracks on concrete surface images},
journal = {Structural Health Monitoring},
volume = {0},
number = {0},
pages = {14759217231173305},
year={2023},
publisher={SAGE Publications Sage UK: London, England},
doi = {10.1177/14759217231173305},
URL = {https://doi.org/10.1177/14759217231173305},
eprint = {https://doi.org/10.1177/14759217231173305},
}
```
