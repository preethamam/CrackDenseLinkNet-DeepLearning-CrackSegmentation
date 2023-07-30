import caffe
caffe.set_mode_cpu()
import numpy as np
from numpy import prod, sum
from pprint import pprint

def print_net_parameters (deploy_file):
    print ("Net: " + deploy_file)
    net = caffe.Net(deploy_file, caffe.TEST)
    print ("Layer-wise parameters: ")
    pprint([(k, v[0].data.shape) for k, v in net.params.items()])
    print ("Total number of parameters: " + str(sum([prod(v[0].data.shape) for k, v in net.params.items()])))
    
deploy_file = "/media/preethamam/Utilities-SSD-2/Xtreme_Programming/Preetham/Deep_Learning/FPHB/examples/fphb/test_fphb_crack.prototxt"
print_net_parameters(deploy_file)