import torch
import torch.nn as nn
import torch.optim as optim
import os

# Specify a path
folderPATH = "/media/preethamam/Utilities-HDD/Liu Best Models/Xtreme_Programming/Liu/project-dlcrack-2d/logs/Linknet/densenet169"
filename = "best_model_iou.pth"
filePATH = os.path.join(folderPATH,filename)

# Load
model = torch.load(filePATH)
model.eval()