import time;
import cv2;
import numpy;
import PIL;
import sys as sys
import math as math

#------------------------------------
#Internal Variables
#------------------------------------
IMAGE_RESOLUTIONS_FORMAT = "UINT8";
TONE_PURE_WHITE = 255;
TONE_PURE_BLACK = 0;

#------------------------------------

#--------------------------------------------------------------------------------------
#This function recieves an image (img) and a threshold (threshold) and sets every tone
#above or equal to the threshold to (tone_above_thresh) and everytone below the threshold to (tone_below_thresh)
#Input 1: img, the image in which the mask will be created
#Input 2: threshold, the threshold to consider
#Input 3: tone_below_thresh, the tone in which the values lower than the threshold will be transformed
#Input 4: tone_above_thresh, the tone in which all values above the threshold will be transformed
#Ouput: The mask
def create_a_mask_from_threshold(img, threshold, tone_below_thresh, tone_above_thresh):

    time_ini = time.time()
    nrows = img.shape[0];
    ncols = img.shape[1];
    ndims = 1;
    
    output_img = img;
    
    #Single component image
    if(ndims == 1):
        for i in range(0, nrows, 1):
            for j in range(0, ncols, 1):
                if (img[i][j] >= threshold):
                    output_img[i][j] = tone_above_thresh;
                else:
                    output_img[i][j] = tone_below_thresh;
                    
    #RGB image (Grayscale nonetheless)                
    elif (ndims == 3):
        for i in range(0, nrows, 1):
            for j in range(0, ncols, 1):
                if (img[i][j][0] >= threshold):
                    output_img[i][j][0] = tone_above_thresh;
                    output_img[i][j][1] = tone_above_thresh;
                    output_img[i][j][2] = tone_above_thresh;
                else:
                    output_img[i][j][0] = tone_below_thresh;
                    output_img[i][j][1] = tone_below_thresh;
                    output_img[i][j][2] = tone_below_thresh;
                    
    str_out = "(Giara Utils) create_a_mask_from_threshold completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return (output_img);
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#internal function
def aux_remove_area(img, labels_areas, area_size_min):
    ncols = img.shape[0];
    nrows = img.shape[1];
    
    result = img;
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            current_label = img[i][j];
            if(labels_areas[current_label] < area_size_min):
                result[i][j] = 0;
    
    return result;
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#Recieves an binary image (image) and an area size (area_size_min)
#and returns a mask with only the interconnected areas that have more pixels than (area_size_min)
def remove_interconnected_areas_below_given_size(idx, image, pixel_color, connection_type, area_size_min):
    time_ini = time.time();
        
    colors_accepted = ["black", "white"];
    
    if(pixel_color == colors_accepted[0]):
        entry_img = image;
    elif(pixel_color == colors_accepted[1]):
        entry_img = (255 - image);
        entry_img = entry_img.astype(numpy.uint8)
        print(entry_img.dtype)

        
    result_label_number, result_label_img = cv2.connectedComponents(entry_img, 8);
    labels_areas = count_all_labels_areas_on_an_image(result_label_img, result_label_number);
    output_label_img = aux_remove_area(result_label_img, labels_areas, area_size_min);
            
    output_image = create_a_mask_from_threshold(output_label_img, 1, 0, 255);

    if(pixel_color == colors_accepted[1]):
        output_image = (255 - output_image);
    
    str_out = "(Giara Utils) remove_interconnected_areas_below_given_size completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return (output_image);
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#Receives a rgb iamge and returns a rgb image with the three channels being the GS channel
def gs_to_rgb(image):
    time_ini = time.time()
    new_image = numpy.zeros((image.shape[1], image.shape[0], 3), dtype=numpy.uint8)
    new_image[:,:,0] =  image[:,:]
    new_image[:,:,1] =  image[:,:]
    new_image[:,:,2] =  image[:,:]          

    str_out = "(Giara Utils) gs_to_rgb completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return new_image
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#This functions counts the number of times than an especific label(pixel tone) appears in an image
#Input 1 (image): int8 or uint8 2D matrix, is the image
#Input 2 (label_to_count): int value, is the pixel tone or the label that will be accounted
#Output (quantity_label): int value, the number of times that the label appears on the image
def count_specific_label_on_image(img, label_to_count):
    time_ini = time.time()
    quantity_label = 0;
    
    ncols = img.shape[0];
    nrows = img.shape[1];
    
    
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            if(img[i][j] == label_to_count):
                quantity_label += 1;
        
    str_out = "(Giara Utils) count_specific_label_on_image completed (Time Taken: %.2f)" % (time.time() - time_ini)
    print(str_out)
    return quantity_label;
    
def count_specific_tone_on_image(image, label_to_count):
    return count_specific_label_on_image(image, label_to_count);
    
def count_specific_value_on_matrix(image, label_to_count):
    return count_specific_label_on_image(image, label_to_count);
#--------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------- 
#Count all label areas on an image
#Input 1: img, the image
#Input 2: num_labels, the label to be considered
#output, the areas of each label present on the img
def count_all_labels_areas_on_an_image(img, num_labels):
    time_ini = time.time()
    ncols = img.shape[0];
    nrows = img.shape[1];
    
    output_areas = numpy.zeros((num_labels+1));
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            output_areas[img[i][j]] += 1;
            
    str_out = "(Giara Utils) count_all_labels_areas_on_an_image completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return output_areas;
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#This function replaces all the appearances of a speficic label or pixel tone to a new one 
#Input 1 (img): int8 or uint8 2D matrix, is the image
#Input 2 (label_to_change): int value, is the pixel tone or the label that will be replaced
#Input 3 (value_to_change): int value, is the pixel tone or the label that will replace the Input 2
#Output: the input image with the Input 1 appearances replaced by Input 2
def change_specific_label_on_image(img, label_to_change, value_to_change):
    time_ini = time.time()
    ncols = img.shape[0];
    nrows = img.shape[1];
    
    result = img;
    
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            if(img[i][j] == label_to_change):
                result[i][j] = value_to_change;
        
    str_out = "(Giara Utils) change_specific_label_on_image completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return result;
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------   
#This function counts the intersection (same cell coordinates) of two different values on two matrices
#Input 1 (img1): int8 or uint8 2D matrix, first image or matrix
#Input 2 (img2): int8 or uint8 2D matrix, second image or matrix
#Input 3 (label1): integer, the reference value of img1
#Input 4 (label2): integer, the reference value of img2
#Output (num_intersections): integer, the number of times that the tw labels appear at the same position in both images 
def count_intersections_of_two_labels_on_image(img1, img2, label1, label2):
    temp_mat = (img1 == label1)
    temp_mat = temp_mat[img2 == label2]
    num_intersections = temp_mat.sum()
    return num_intersections;

def count_intersections_of_two_tones_on_image(img1, img2, label1, label2):
    return count_intersections_of_two_labels_on_image(img1, img2, label1, label2);
    
def count_intersections_of_two_labels_on_matrices(img1, img2, label1, label2):
    return count_intersections_of_two_labels_on_image(img1, img2, label1, label2);
    
def count_intersections_of_two_values_on_matrices(img1, img2, label1, label2):
    return count_intersections_of_two_labels_on_image(img1, img2, label1, label2);
#--------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------
#Input 1 (img1): int8 or uint8 2D matrix, first image or matrix
def create_a_mask_for_specific_tone(img, tone):
    time_ini = time.time()
    ncols = img.shape[0];
    nrows = img.shape[1];
 
    result = img;
    result = result.astype(numpy.uint8)
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            if(img[i][j] == tone):
                result[i][j] = TONE_PURE_WHITE;
            else: 
                result[i][j] = TONE_PURE_BLACK;
               
    str_out = "(Giara Utils) create_a_mask_for_specific_tone completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return result;
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#Recieves an image, a labelled image and a label
#and return the mean pixel intensity of the areas marked with the (label_to_consider)
#Input 1: imgimg, the original image in which the intensity will be calculated
#Input 2: labeled_img, the labelled areas of img (each region has a label)
#input 3: label_to_consider, the label in (labeled_img) that will be considered
#Output: The mean intensity of the area marked by <get_mean_intensity_of_label_area>
def get_mean_intensity_of_label_area(img, labeled_img, label_to_consider):
    time_ini = time.time()
    mean_intensity = 0;
    num_pixels = 0;
    sum_of_values = 0;
    
    ncols = img.shape[0];
    nrows = img.shape[1];
    
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            if(labeled_img[i][j] == label_to_consider):
                num_pixels += 1;
                sum_of_values += img[i][j];
                
    if(num_pixels > 0):
        mean_intensity = sum_of_values / num_pixels;
    else:
        mean_intensity = 0;
    str_out = "(Giara Utils) get_mean_intensity_of_label_area completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return mean_intensity;
#--------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------
def print_matrix_to_file(matrix_in, file_name):
    mat = numpy.matrix(matrix_in);
    with open(file_name,'wb') as f:
        for line in mat:
            numpy.savetxt(f, line, fmt='%.2f');
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
#The bsxfun funtion from matlab
#performs pixel-wise opperations on two matrices
#Input 1: the operation to be performed (["times", "plus", "minus", "or", "and", "rdivide"])
#Input 2 and 3: The two matrices
#Output: The result of the operation
def bsxfun(function, matrix1, matrix2):
    time_ini = time.time()
    
    allowed_functions = ["times", "plus", "minus", "or", "and", "rdivide"]
    
    #For Two Dimensional array inputs    
    if(len(matrix1.shape) == 2):
        output_mat = numpy.zeros((matrix1.shape[0], matrix1.shape[1]), dtype=numpy.uint8)
        #Performing the operation for two arrays
        if(not numpy.isscalar(matrix2)):
            if((matrix1.shape != matrix2.shape)):
                print("ERROR (bsxfun function): The two arrays must have the same dimensionality")
                print("Array 1 infos:")
                print_array_dimensions(matrix1)
                print("Array 2 infos::")
                print_array_dimensions(matrix2)
                #sys.exit()
                
            if(function == allowed_functions[0]): #times
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] * matrix2[i][j]
                        
            elif(function == allowed_functions[1]): #plus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] + matrix2[i][j]
                        
            elif(function == allowed_functions[2]): #minus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] - matrix2[i][j]
                        
            elif(function == allowed_functions[3]): #or
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = numpy.uint8(bool(matrix1[i][j]) or bool(matrix2[i][j]))
                        
            elif(function == allowed_functions[4]): #and
                for i in range(matrix1.shape[1]):
                    for j in range(matrix2.shape[0]):
                        output_mat[i][j] = numpy.uint8(bool(matrix1[i][j]) and bool(matrix2[i][j]))
                        
            elif(function == allowed_functions[5]): #rdivide
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] / matrix2[i][j]
                        
                        
        #Performing the Operation to one array and one scarlar 
        else:
            scalar_var = matrix2
            if(function == allowed_functions[0]): #times
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] * scalar_var
                        
            elif(function == allowed_functions[1]): #plus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] + scalar_var
                        
            elif(function == allowed_functions[2]): #minus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] - scalar_var
                        
            elif(function == allowed_functions[3]): #or
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = numpy.uint8(bool(matrix1[i][j]) or bool(scalar_var))
                        
            elif(function == allowed_functions[4]): #and
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = numpy.uint8(bool(matrix1[i][j]) and bool(scalar_var))
                        
            elif(function == allowed_functions[5]): #rdivide
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        output_mat[i][j] = matrix1[i][j] / scalar_var
                        
    #If matrix1 is a 3D matrix
    elif(len(matrix1.shape) == 3):
        output_mat = numpy.zeros((matrix1.shape[0], matrix1.shape[1], matrix1.shape[2]), dtype=numpy.uint8)
        #Performing the operation for two arrays
        if(not numpy.isscalar(matrix2)):
            if((matrix1.shape != matrix2.shape)):
                print("ERROR (bsxfun function): The two arrays must have the same dimensionality")
                print("Array 1 infos:")
                print_array_dimensions(matrix1)
                print("Array 2 infos::")
                print_array_dimensions(matrix2)
                #sys.exit()
                
            if(function == allowed_functions[0]): #times
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k]= matrix1[i][j][k] * matrix2[i][j][k]
                        
            elif(function == allowed_functions[1]): #plus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] + matrix2[i][j][k]
                        
            elif(function == allowed_functions[2]): #minus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] - matrix2[i][j][k]
                        
            elif(function == allowed_functions[3]): #or
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = numpy.uint8(bool(matrix1[i][j][k]) or bool(matrix2[i][j][k]))
                        
            elif(function == allowed_functions[4]): #and
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = numpy.uint8(bool(matrix1[i][j][k]) and bool(matrix2[i][j][k]))
                        
            elif(function == allowed_functions[5]): #rdivide
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] / matrix2[i][j][k]
                        
                        
        #Performing the Operation to one array and one scarlar 
        else:
            scalar_var = matrix2
            if(function == allowed_functions[0]): #times
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] * scalar_var
                        
            elif(function == allowed_functions[1]): #plus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] + scalar_var
                        
            elif(function == allowed_functions[2]): #minus
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] - scalar_var
                        
            elif(function == allowed_functions[3]): #or
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = numpy.uint8(bool(matrix1[i][j][k]) or bool(scalar_var))
                        
            elif(function == allowed_functions[4]): #and
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = numpy.uint8(bool(matrix1[i][j][k]) and bool(scalar_var))
                        
            elif(function == allowed_functions[5]): #rdivide
                for i in range(matrix1.shape[1]):
                    for j in range(matrix1.shape[0]):
                        for k in range(matrix1.shape[2]):
                            output_mat[i][j][k] = matrix1[i][j][k] / scalar_var
                        
                        
    str_out = "(Giara Utils) bsxfun completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return output_mat
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
def print_array_dimensions(array):
    num_dims = len(array.shape)
    print("Array has %d dimensions" % num_dims)
    for i in range(num_dims):
        print("Dimension %d has size of %d" % (i+1, array.shape[i]))
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
def convert_uint16_to_uint8(img):
    converted_img = numpy.zeros(img.shape, dtype=numpy.uint8)
    
    if(len(img.shape) == 2):
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                converted_img[i][j] =  img[i][j] / 255
    elif(len(img.shape) == 3):
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                for k in range(img.shape[2]):
                    converted_img[i][j][k] =  img[i][j][k] / 255
                    
    return converted_img

#--------------------------------------------------------------------------------------
def rosin_threshold_v1(histogram_to_process):

    #Step 01: Finding the hitogram peak
    max_value = max(histogram_to_process)
    max_hist_peak_index = numpy.argmax(histogram_to_process)

    #Step 02: Finding the first empty bin after the last filled bin
    vet_of_non_zero_index = numpy.nonzero(histogram_to_process)
    vet_of_non_zero_index = vet_of_non_zero_index[0]
    if(vet_of_non_zero_index[-1] == (len(histogram_to_process) - 1)):
        first_empty_bin_index = len(histogram_to_process - 1)
    else:
        first_empty_bin_index = (vet_of_non_zero_index[-1] + 1)
        

    #Step 3: Finding the maximum distance beteween the line S01-S02 and the
    #histogram bin
    #y = ax + b
    greatest_dist_index = max_hist_peak_index;
    greatest_dist_value = -1; 
    
    #The line from S91 to S02 is: L01: y = ax + b
    a_value = float(0 - float(histogram_to_process[max_hist_peak_index])); #%ok
    a_value = a_value / (float(first_empty_bin_index) - float(max_hist_peak_index)); #ok
    b_value = float(histogram_to_process[max_hist_peak_index]) - a_value*float(max_hist_peak_index)

    perpend_a = float(-1) / a_value;
    
    for ii in range(max_hist_peak_index, first_empty_bin_index, 1):
        #Perpendicular line LPER to L01 from the current bin
        perpend_x_ini = ii; #ok
        perpend_y_ini = histogram_to_process[ii]; #ok
        perpend_b = float(perpend_y_ini) - perpend_a*float(perpend_x_ini); #ok

        #perperndicular point in L01
        x_L01 = (float(b_value)- float(perpend_b));
        x_L01 = x_L01 / (float(perpend_a) - float(a_value));
        y_L01 = a_value*x_L01 + b_value;

        #Euclidean Distance
        current_dist = (float(y_L01)-float(perpend_y_ini))**2
        current_dist = current_dist + ((float(x_L01)-float(perpend_x_ini))**2)
        current_dist = math.sqrt(current_dist);

       #Comparing Distances
        if(current_dist > greatest_dist_value):
            greatest_dist_value = current_dist;
            greatest_dist_index = ii;

    defined_threshold = greatest_dist_index;
    
    return defined_threshold
#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
def use_labels_to_change_mask_intensity(img, labels, label_to_change, new_intensity):
    nrows = labels.shape[1]
    ncols = labels.shape[0]
    mask_output = img;
    mask_output = mask_output.astype(numpy.uint8)
    
    for i in range(nrows):
        for j in range (ncols):
            if(labels[i][j] == label_to_change):
                mask_output[i][j] = new_intensity

    return mask_output
#--------------------------------------------------------------------------------------