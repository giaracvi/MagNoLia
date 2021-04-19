from PIL import Image
import glob
import numpy as np 
import cv2
import os
import time as t
import matplotlib.pyplot as plt


def DAPI_Segmentation(exp):
    path_in = exp.path_DAPI
    path_out = exp.path_DAPI_out
    min_body_size_DAPI = exp.DAPI_min_body_size
    image_format = exp.image_format
    num_images_to_process = exp.num_images_to_process
    
    
    flag_success = 1;
    K = exp.DAPI_number_of_clusters;
    cont = 0;

    #Reading the input images
    image_list = glob.glob(path_in + '/*' + image_format);
    
    if(num_images_to_process != 0):
        image_list = image_list[0:num_images_to_process]
        
    num_images_to_process = len(image_list)
          
    #Creating output dir, if not exists
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0);
    
    #------------------------------------------
    for imagen in image_list:
            execution_time_ini=t.time();
            
            #name
            imagen=imagen.replace('\\','/');
            Nombre = imagen;
            only_image_name = imagen[imagen.rfind('/')+1:imagen.rfind('.')];

            #Read image
            img = cv2.imread(Nombre,0)#cv2.IMREAD_GRAYSCALE)
        
            #Reshape
            Z = img.reshape((-1,1))
            Z = np.float32(Z)
        
            ##Apply kmeans()
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
             
            # Now convert back into uint8, and make original image
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img.shape))  
            
            image_binary=np.zeros(np.shape(res2),np.uint8)
            valores=np.unique(res2)
            image_binary[res2==valores[1]]=255
            image_binary[res2==valores[2]]=255
        
            
            #Dilation
            D1=2
            kernel = np.ones((D1,D1), np.uint8) 
            img_dilation = cv2.dilate(image_binary, kernel, iterations=1)             
            
            #Erosion
            E1=2
            kernel = np.ones((E1,E1), np.uint8) 
            img_erosion = cv2.erode(img_dilation, kernel, iterations=2)             

            #Eliminate empty
            n, mask=cv2.connectedComponents(img_erosion)
            contours_lim=[]
            
            for label in range(1,n):
                img_canvas=np.zeros([np.shape(img)[0],np.shape(img)[1]],dtype=np.uint8)
                img_canvas[mask==label]=255
                
                points=sum(sum(mask==label))
                
                #Size and shape study
                contours,hierarchy = cv2.findContours(img_canvas, 1, 2)[-2:]
                
                for cnt in contours:
                
                    area = cv2.contourArea(cnt)
                    
                    if area >= exp.DAPI_min_body_size and points>(area*0.8):
                        contours_lim.append(cnt)
                    
            img_contours=np.ones([np.shape(img)[0],np.shape(img)[1]])
            cv2.drawContours(img_contours, contours_lim, -1, (0,255,0), thickness=cv2.FILLED)   
            cont+=1
            
            cv2.imwrite(path_out+"/"+only_image_name+'.png',img_contours*255)
            time_taken = t.time()-execution_time_ini;
            str_out = "(DAPI Segmentation): Image " + str(cont) + " of " + str(num_images_to_process) + "(" + str(float("{:.2f}".format(time_taken))) + 's)';
            print(str_out);
    #------------------------------------------
    return flag_success;
    
    
def print_image_names(image_vector):
    
    for image in image_vector:
        print(image);        