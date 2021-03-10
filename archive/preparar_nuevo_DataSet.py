import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np 
import json
import cv2
import os


train_csv_path = "/home/santiago/Documentos/yolov5/yolov5/archive/new_train.csv"
new_train_csv_path = "/home/santiago/Documentos/yolov5/yolov5/archive/nuevo_train.csv"
new_csv  = pd.DataFrame()
train_csv = pd.read_csv(train_csv_path)

def crear_carpetas(carpeta):
    if not os.path.exists(carpeta):
        os.mkdir(carpeta)


# create the file structure
def create_root(file_name, width_value, height_value):
    root = ET.Element('annotation')
    folder = ET.SubElement(root, 'folder');                     folder.text = "images"
    filename = ET.SubElement(root, 'filename');                  filename.text = str(file_name)
    segmented = ET.SubElement(root, 'segmented');               segmented.text = "0"


    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width');                       width.text = str(width_value)
    height = ET.SubElement(size, 'height');                     height.text = str(height_value)
    depth = ET.SubElement(size, 'depth');                       depth.text = "3"

    return root

def create_object(root, class_obj, xmin_value, xmax_value, ymin_value, ymax_value):

    object_anotation = ET.SubElement(root, 'object')
    name = ET.SubElement(object_anotation, 'name');             name.text = str(class_obj)
    pose = ET.SubElement(object_anotation, 'pose');             pose.text = "Unspecified"
    truncated = ET.SubElement(object_anotation, 'truncated');   truncated.text = "0"
    occluded = ET.SubElement(object_anotation, 'occluded');     occluded.text = "0"
    difficult = ET.SubElement(object_anotation, 'difficult');   difficult.text = "0"


    bndbox = ET.SubElement(object_anotation, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin');                       xmin.text = str(xmin_value)
    ymin = ET.SubElement(bndbox, 'ymin');                       ymin.text = str(ymin_value)
    xmax = ET.SubElement(bndbox, 'xmax');                       xmax.text = str(xmax_value)
    ymax = ET.SubElement(bndbox, 'ymax');                       ymax.text = str(ymax_value)

    return root


allowed_classes = ["face_with_mask", "face_no_mask"]

def annotations_txt2xml(file_name, annotations_path, images_path, new_anotations_path,new_image_path,train_csv):
    name_image = file_name.split("/")
    name_image = name_image[1]
    im_path = os.path.join(images_path, file_name)
    ano_path = annotations_path
    new_ano_path = os.path.join(new_anotations_path, name_image.split(".")[0]+".xml")


    assert os.path.exists(im_path), "No existe el archivo " + im_path
    assert os.path.exists(ano_path), "No existe el archivo: " + ano_path
    
    labels_file = open(ano_path,'r')
    labels_lines = labels_file.readlines()
    is_image = False    
    count = -1
    #Carga de imagen para las dimen
    img = cv2.imread(im_path)
    shape_img = img.shape
    cv2.imwrite(new_image_path+name_image, img) 
    #DireRoot
    root = create_root(name_image, height_value= shape_img[0], width_value=shape_img[1])
    count_occlusion = 0
    #Clase siempre sin mascara
    class_name = allowed_classes[1]
    for line in labels_lines:            
        if (count > 0):
            new_label = line.strip().split(" ")  
            #print(new_label[-2] )
            if (new_label[-2] == str(0) or new_label[-2] == str(1)):
                
                count_occlusion += 1
                x2 = int(new_label[0]) + int(new_label[2])
                y2 = int(new_label[1]) + int(new_label[3])
                x_left,y_down,x_right,y_up = new_label[0],new_label[1], x2 , y2
                train_csv = train_csv.append({'nuevo_name': name_image,'x1':x_left,'x2':x_right,'y1':y_down,'y2':y_up,'classname':class_name},ignore_index=True)
                #train_csv = new_csv
                root = create_object(root, class_name, xmin_value=x_left, xmax_value=x_right, ymin_value=y_down, ymax_value=y_up)            
            count -=1
        if(is_image):
            count = int(line)
            is_image = False

        if(name_image in line):
            is_image = True
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(new_ano_path, "w") as f:
       f.write(xmlstr)
    return train_csv,count_occlusion



new_anotations_path = "/home/santiago/Documentos/yolov5/yolov5/archive/final_annotations/"
labels_path = "/home/santiago/Documentos/yolov5/yolov5/archive/newTrain/wider_face_split/wider_face_train_bbx_gt.txt"
path = "/home/santiago/Documentos/yolov5/yolov5/archive/newTrain/WIDER_train/images/"
name_image = "0--Parade/0_Parade_Parade_0_476.jpg"
path_prueba_imagen = path+name_image


new_images_path = "/home/santiago/Documentos/yolov5/yolov5/archive/final_images/"
crear_carpetas(new_anotations_path)
crear_carpetas(new_images_path)


images_names = pd.read_csv('/home/santiago/Documentos/yolov5/yolov5/archive/nombre_imagenes_nuevo_dataset.csv').values
print("ANTES ",len(train_csv))
contador = 0
for indx, im_name in enumerate(images_names):
    train_csv, count = annotations_txt2xml(im_name[0],annotations_path=labels_path,images_path= path,new_anotations_path = new_anotations_path,new_image_path=new_images_path,train_csv=train_csv)
    contador +=count
    
    if indx % 500==0:        
        print("Cantidad de imágenes procesadas: ", indx)
        print(".")
        print(".")
        print(".")
print(contador)
print("DESPUES ",len(train_csv))

train_csv.to_csv(new_train_csv_path)



    