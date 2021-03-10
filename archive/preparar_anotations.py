
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
import numpy as np 
import json
import cv2
import os
from shutil import copyfile
import shutil


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



def crear_carpetas(carpeta):
    if not os.path.exists(carpeta):
        os.mkdir(carpeta)
    
def get_nuevos_nombres(df):
    nuevos_nombres = df.name.copy()
    print("asdASD")

    images = df.name.unique()
    images_sin = [image.split(".")[0] for image in images]
    seen = ['6101.jpg', '6166.jpg', '6108.jpg', '6137.jpg', '6200.jpg', '6100.jpg', '6107.jpg']#set([x for x in images_sin if images_sin.count(x) > 1])
    #print(images_sin )
    imagenes_re_for = ['6101.jpg', '6166.jpg', '6108.jpg', '6137.jpg', '6200.jpg', '6100.jpg', '6107.jpg']#[image for image in images if image.split(".")[0] in seen]
    
    for indx, image in enumerate(imagenes_re_for):
        #print("2. ASDASD")
        nombre, ext = image.split(".")
        particular_Anotation = path_annotations + "/" + image +".json"        
        new_name = nombre+"_"+str(indx)+"." + ext
        shutil.copy(particular_Anotation, (path_annotations + "/"+new_name+".json"))
        #print(nombre+"_"+str(indx)+"." + ext )
        nuevos_nombres[df.name == image] = new_name
    print("2. finalASDASD")
    df["nuevo_name"] = nuevos_nombres

    
    

    return df

def plotear_imagen(img,bbox):
    plotted_img = draw_rect(img, bbox)
    plt.imshow(plotted_img)
    plt.show()
    
def augment_data(image_path, new_image_path, anotations_path, new_anotations_path):
    with open(anotations_path) as json_file:
        
        data = json.load(json_file)
        
        file_name = data["FileName"]
        #print(file_name)
        #print(new_image_path)
        img = cv2.imread(image_path)
        shape_img = img.shape
        _ , nuevaImagen = new_image_path.split("/")
        root = create_root(nuevaImagen, height_value= shape_img[0], width_value=shape_img[1])
        bboxes=[]
        if(len(data["Annotations"])<1):
            print("NO HAY ANOTACIONES",file_name)

        for anot in data["Annotations"]:
            
            class_name = anot["classname"]
            if class_name == "face_with_mask_incorrect":
                class_name = "face_no_mask"
            #print("ENTró", class_name)
            if class_name in allowed_classes:
                #print("ENTROOOO")
                x_left,y_down,x_right,y_up = anot["BoundingBox"]
                bboxes.append([x_left,y_down,x_right,y_up,np.where(allowed_classes == class_name)[0][0]])
        
        
        if len(bboxes)>0:

            for bbox in bboxes:
                x_left,y_down,x_right,y_up = bbox[:-1]
                indx = bbox[-1]
                class_name = allowed_classes[indx]
                root = create_object(root, class_name, xmin_value=x_left, xmax_value=x_right, ymin_value=y_down, ymax_value=y_up)
            
            cv2.imwrite(new_image_path, img) 
        else:
            cv2.imwrite(new_image_path, img) 
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(new_anotations_path, "w") as f:        
        f.write(xmlstr)



new_path_annotations = os.path.join("final_annotations")
new_images = os.path.join("final_images")
path_annotations = os.path.join("annotations")
path_images = os.path.join("images")
path_csv = "train.csv"
new_path_csv = "new_train.csv"


crear_carpetas(new_images)
crear_carpetas(new_path_annotations)

allowed_classes = np.array(["face_with_mask", "face_no_mask"])



df = pd.read_csv(path_csv)


if (not os.path.exists(new_path_annotations)):
    os.mkdir(new_path_annotations)
    assert os.path.exists(new_path_annotations), "No existe el nuevo directorio"


print("DF orig", len(df))
df = get_nuevos_nombres(df)
df.to_csv(new_path_csv,index=False)
seen = ['6101_0.jpg', '6166_1.jpg', '6108_2.jpg', '6137_3.jpg', '6200_4.jpg', '6100_5.jpg', '6107_6.jpg']
#print(df.loc[df['nuevo_name']=='6107_6.jpg'])
for indx, name in enumerate(seen):
    name_a,ext = name.split(".")
    #print(df.name[df.name == name_a].unique())


        


try:
    for indx, imagen_new_path in enumerate(df.nuevo_name.unique()):
        
        path_orig = df.name[df.nuevo_name == imagen_new_path].unique()[0]
        #print(path_orig)
        #print(imagen_new_path)
        imagen_path_original = os.path.join(path_images, path_orig)
        imagen_path_final = os.path.join(new_images, imagen_new_path)
        #print(imagen_path_original)
        #print(imagen_path_final)
        anotations_path_original = os.path.join(path_annotations, path_orig+".json")
        anotations_path_final = os.path.join(new_path_annotations, imagen_new_path.split(".")[0]+".xml")

        augment_data(image_path = imagen_path_original, new_image_path = imagen_path_final, anotations_path = anotations_path_original, new_anotations_path = anotations_path_final)

        if indx % 500==0:
            print("Cantidad de imágenes procesadas: ", indx)
            print(".")
            print(".")
            print(".")
 
except Exception as e:
    print(imagen_new_path)
    print(e)
    raise(e)


print("TERMINÓ agument")

df.to_csv(new_path_csv,index=False)

