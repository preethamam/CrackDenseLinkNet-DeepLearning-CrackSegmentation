import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from model import *
import skimage.transform as trans
import tensorflow as tf
import skimage.io as io
import pickle
import torch
import time
# LD_LIBRARY_PATH = '/home/username/miniconda3/envs/vision/lib/'
#model_name = './trained/4/ImprovedFCN_40'
model_name = './ImprovedFCN'
#model_name = './ImprovedFCN_origin'
model = unet3s2()
model.load_weights(model_name + '.hdf5')
#CrackSegNet
#test_path="../test_data/CrackSegNet/TestingCracks"
#test_path = "../test_data/DS3/TestingCracks"
test_path = "/media/preethamam/Utilities-SSD-1/Xtreme_Programming/ZZZ_Data/DLCrack/Liu+Xincong+DS3+CrackSegNet/Testing/DS3/TestingCracks"
data_name = os.listdir(test_path)
total_time = 0

softmax_dict = {}
for i in data_name:
   if 'predict' not in i:
        img = io.imread(os.path.join(test_path, i)) 
        size_x = img.shape[0]
        size_y = img.shape[1]
        img = trans.resize(img, (512, 512))
        img = np.reshape(img, (1,) + img.shape)

        start_time = time.time()
        results = model.predict(img, verbose=1)
        predict_time = time.time() - start_time
        total_time += predict_time

        # results = np.squeeze(results)
        # results[results <= 0.5] = 0
        # results[results > 0.5] = 1
        prediction = torch.reshape(torch.from_numpy(results), (1, 1, 512, 512))  # convert to pytorch tensor
        m = torch.nn.Upsample(size=(size_x,size_y), mode="nearest")  # upsampling
        resized_prediction = m(prediction).numpy()  # upsampling
        softmax_dict[i] = list(np.ndarray.flatten(resized_prediction))  # save flattened to dictionary
        filename = i.replace('.jpg','.png')
        # io.imsave(os.path.join("./ExperimentResults/pred", filename+".png"), results)

print(len(data_name))
print(total_time)
print("Average time: " + str(total_time/len(data_name)))
print("============>>>> Finish predict ... <<<<============")

with open('soft_max_output.pickle', 'wb') as handle:
    pickle.dump(softmax_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
