# ia2-proyecto

![Banner](images/banner.png)

Se recomienda el uso de docker para la ejecución del proyecto, con la imagen de Pytorch y uso de cuda 11.1 si se ejecuta sobre una GPU NVIDIA

<code>docker pull pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel</code>

Para la instalación de Docker y uso de la GPU en los contenedores puede seguir el siguiente tutorial : https://www.tensorflow.org/install/docker

Una vez instalado docker y se haya comprobado su correcto funcionamiento, debe crear el contenedor de ls siguiente manera:

<code>
docker run --gpus all -p 8888:8888 -p 6006:6006 -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro  --rm -v {PATH MAQUINA HOST}:/workspace pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel 
</code>

es recomendable tener un directorio compartido entre su maquina host y el contenedor, si no desea crear este espacio compartido debe remover las lineas <code>-v {PATH MAQUINA HOST}:/workspace</code>. 

##
<b>Entrenamiento:</b>
Para la ejecución del entrenamiento, estando en la consola del contenedor, ejecute:

<code>pip install tensorboard</code>

<code>python train.py --img 320 --batch 16 --epochs 300 --data data/facemask.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt</code>

##
<b>Detección</b>
Una vez terminado el entrenamiento, desde la consola del contenedor ejecute: 

<code>cp runs/train/exp/weights/best.pt weights</code>

<code>python detect.py --source Dataset/images/test --img-size 320 --conf 0.4 --weights weights/best.pt</code>







