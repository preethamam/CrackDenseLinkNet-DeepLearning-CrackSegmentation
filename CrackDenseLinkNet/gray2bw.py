import os
import cv2
a = 1
methods = ['./CrackSegNet', './DS3', './Liu', './Xincong']
for m in methods:
    for dirname, _, filenames in os.walk(m+'/pred_40k'):
        for filename in filenames:
            print(a)
            a += 1
            image_file = os.path.join(dirname,filename)
            image = cv2.imread(image_file)
            _,image = cv2.threshold(image,127.5,255,cv2.THRESH_BINARY)
            cv2.imwrite(image_file,image)
