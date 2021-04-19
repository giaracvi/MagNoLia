import glob;
import os;
import cv2;
import time;
import giara_utils;
import matplotlib as matplotlib
import numpy 

def ChAT_Segmentation(exp):
    path_in = exp.path_ChAT
    path_out = exp.path_ChAT_out
    min_body_size_C3 = exp.ChAT_min_body_size
    image_format = exp.image_format
    num_images_to_process = exp.num_images_to_process
    flag_success = 1;
    connection_type = 4;
    
    str_to_search = path_in + '/*' + image_format

    image_list = glob.glob(str_to_search);
    num_images = len(image_list);

    if(num_images_to_process != 0):
        image_list = image_list[0:num_images_to_process]
        num_images = num_images_to_process
        
    #Creating output dir, if not exists
    if not os.path.exists(path_out):
        os.makedirs(path_out)
     
    for idx, image in enumerate(image_list):
        execution_time_ini = time.time();
        image_name_orig = image.replace('\\','/');
        only_image_name = image_name_orig[image_name_orig.rfind('/')+1:image_name_orig.rfind('.')];
        print("(ChAT Segmentation): Image " + str(idx+1) + " of " + str(num_images) + (" (") + only_image_name + ")");
        
        #Read image
        time_ini = time.time();
        img = cv2.imread(image_name_orig, 0); 
        
        #Get Threshold
        img_hist = numpy.histogram(img, bins = 256, range=(0, 255))
        img_hist = img_hist[0]
        threshold_value = giara_utils.rosin_threshold_v1(img_hist)
        
        #Evaluate blob size
        masked_img = giara_utils.create_a_mask_from_threshold(img, threshold_value, 0, 255);
        masked_img2 = giara_utils.remove_interconnected_areas_below_given_size(idx, img, "black", connection_type, min_body_size_C3);
        masked_img3 = giara_utils.remove_interconnected_areas_below_given_size(idx, masked_img2, "white", connection_type, min_body_size_C3);
        cv2.imwrite((path_out + "/" + only_image_name + "." + image_format), masked_img3);
        
        time_taken = time.time()-execution_time_ini;
        str_out = "--Time Taken: " + str(float("{:.2f}".format(time_taken))) + 's';
        print(str_out);
        
    return flag_success;