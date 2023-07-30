import os
import tensorflow as tf
import skimage.io as io
import pickle
import torch
import numpy as np


test_path = "./CrackSegNet"
data_name = os.listdir(test_path)

softmax_dict = {}
for i in data_name:
   if 'predict' not in i:
        img = io.imread(os.path.join(test_path, i)) 
        img = img/255
        filename = i.replace('_gf','')
        softmax_dict[filename] = list(np.ndarray.flatten(img))
        # io.imsave(os.path.join("./ExperimentResults/pred", filename+".png"), results)
with open('soft_max_output.pickle', 'wb') as handle:
    pickle.dump(softmax_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)