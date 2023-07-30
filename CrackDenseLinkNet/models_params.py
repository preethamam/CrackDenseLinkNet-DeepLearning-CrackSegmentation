from torchvision import models
from torchsummary import summary

model_vgg16 = models.resnet101()
summary(model_vgg16.cuda(),(3, 224, 224))