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
<p align="justify">
In this study, four datasets were used to evaluate the crack segmentation capability of five CNN-based semantic segmentation networks. Datasets FCN [1], DeepCrack [2], and CrackSegNet [3] are available to public. In addition, the proposed method dataset CrackDenseLinkNet is currently private (it will be made public during the publication) and created around the University of Southern California campus. The complexity of images varies gradually from CrackDenseLinkNet, FCN, DeepCrack, and CrackSegNet. Figure 2 shows the four datasets’ sample images. 

CrackDenseLinkNet dataset consists of 250 testing images of a concrete surface. In this dataset, the cracks have a stronger contrast, texture, and wider cracks relative to the other datasets. About 249 images have high textural content out of 250. The FCN dataset is primarily of the concrete surface and consists of 154 testing crack samples that vary in size, out of which 6 and 148 images have low and high texture, respectively. In this dataset, crack widths are thin and thicker. Furthermore, it consists of longitudinal, transverse, and surface cracks. All the images are of high quality. However, it was observed that some of the groundtruth images were wrongly labeled (labels are thicker than the actual width of the cracks). 

Similar to the FCN dataset, the DeepCrack dataset consists of 237 testing samples of various longitudinal, transverse, and surface cracks. About 78 and 22% of the images have concrete and asphalt material surfaces, respectively. The crack width varies largely across the images of this dataset. Around 21 and 216 images in this dataset have low and high textures, respectively. Lastly, CrackSegNet consists of 184 testing images of the concrete surface, 5 and 179 images of low and high texture, respectively. The crack width of this dataset is relatively thinner compared to the other datasets. This is the most challenging dataset to train the semantic segmentation CNN methods, as it contains highly blurry images, textural noise, and paint artifacts. This dataset was purposefully included to assess the limitations of all five methods in comparison.
</p>

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

## References
<a id="1">[1]</a> Yang X, Li H, Yu Y, et al. Automatic pixel-level crack detection and measurement using fully convolutional network. Comput-Aided Civil Infrastruct Eng 2018; 33(12): 1090–1109.

<a id="2">[2]</a> Liu Y, Yao J, Lu X, et al. Deepcrack: a deep hierarchical feature learning architecture for crack segmentation. Neurocomputing 2019; 338: 139–153.

<a id="3">[3]</a> Ren Y, Huang J, Hong Z, et al. Image-based concrete crack detection in tunnels using deep fully convolutional networks. Constr Build Mater 2020; 234: 117367.
