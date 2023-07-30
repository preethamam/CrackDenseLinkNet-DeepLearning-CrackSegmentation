import os
import numpy as np
import shutil
import random

np.random.seed(1)

# # Creating Train / Val / Test folders (One time use)
root_dir = './CFD'
gt = '/GroundTruth'
img = '/Cracks'

# Creating partitions of the data after shuffeling
currentCls = gt
src = root_dir + currentCls # Folder to copy images from

train_save_dir = '/TrainingCracksGroundTruth' if currentCls == gt else '/TrainingCracks'
val_save_dir = '/ValidationCracksGroundTruth' if currentCls == gt else '/ValidationCracks'
test_save_dir = '/TestingCracksGroundTruth' if currentCls == gt else '/TestingCracks'

os.makedirs(root_dir + train_save_dir)
os.makedirs(root_dir + val_save_dir)
os.makedirs(root_dir + test_save_dir)

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.5), int(len(allFileNames)*0.6)])

train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, root_dir + train_save_dir)

for name in val_FileNames:
    shutil.copy(name, root_dir + val_save_dir)

for name in test_FileNames:
    shutil.copy(name, root_dir + test_save_dir)
