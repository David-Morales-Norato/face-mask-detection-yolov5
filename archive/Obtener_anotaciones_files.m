clear; clc; 
load('wider_face_train.mat', "file_list", "face_bbx_list", "event_list");



cantidad_de_imagenes = 400;
cant_generar_por_carpeta = ceil(cantidad_de_imagenes/length(file_list));

imagenes_path = strings(cantidad_de_imagenes,1);
folder_path = strings(cantidad_de_imagenes,1);
for u = 0:length( file_list)-1
    archivos_carp = file_list{u+1};
    indx = randi(length(archivos_carp), cant_generar_por_carpeta,1);
    disp(['min ' num2str(cant_generar_por_carpeta*u) ' max ' num2str((u+1)*cant_generar_por_carpeta)])
    for i = 1:length(indx)
        imagenes_path(cant_generar_por_carpeta*u+i) = archivos_carp{indx(i)};
        folder_path(cant_generar_por_carpeta*u+1:(u+1)*cant_generar_por_carpeta+1) = event_list{u+1};
    end
end

csv_final  = strings(cantidad_de_imagenes,1);
for u = 1:cantidad_de_imagenes
    csv_final(u) = folder_path(u)+"/"+imagenes_path(u)+".jpg";
end
writematrix(csv_final,'nombre_imagenes_nuevo_dataset.csv') 
