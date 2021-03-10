import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import os
import json
from data_aug.data_aug import *
from data_aug.bbox_util import *
import pandas as pd
from xml.dom import minidom

def read_image(impath):
    img = cv2.imread(impath)
    return img


def get_bboxes(im_name):
    doc = minidom.parse(im_name)
    bboxes = []

    for obj in doc.getElementsByTagName("object"):
        x_left = obj.getElementsByTagName("bndbox")[0].getElementsByTagName("xmin")[0].childNodes[0].nodeValue
        y_down = obj.getElementsByTagName("bndbox")[0].getElementsByTagName("ymin")[0].childNodes[0].nodeValue
        x_right = obj.getElementsByTagName("bndbox")[0].getElementsByTagName("xmax")[0].childNodes[0].nodeValue
        y_up = obj.getElementsByTagName("bndbox")[0].getElementsByTagName("ymax")[0].childNodes[0].nodeValue
        bboxes.append([x_left,y_down,x_right,y_up,0])
    return bboxes

df = pd.read_csv("new_train.csv")
images_path = "final_images"

images = df.nuevo_name.unique()

file_ano = "final_annotations/6418.xml"
img = read_image(os.path.join(images_path,"6418.png"))
bboxes = get_bboxes(file_ano)
bboxes = np.array(bboxes,dtype ="float64")
plotted_img = draw_rect(img, bboxes)

plt.imshow(plotted_img)
plt.show()
# for image in np.random.choice(images int(0.1*len(images))):
#     nombre = images[i]

#     img = read_image(os.path.join(images_path, nombre))

#     bboxes = get_bboxes(nombre, df = df)
    
#     draw_rect(img, bboxes)