import os
import cv2
for dirname, _, filenames in os.walk('./CrackSegNet_resize'):
    for filename in filenames:
        image_file = os.path.join(dirname,filename)
        image = cv2.imread(image_file,0)
        image = cv2.bitwise_not(image)
        cv2.imwrite(image_file,image)

