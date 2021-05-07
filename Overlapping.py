import glob
import os
import cv2 as cv2
import time
import giara_utils as giara_utils
import magnolia_utils as magnolia_utils
import numpy as numpy

def Overlapping(exp):
    
    #--------------------------------------------------------
    #naming the variables to use internally
    path_DAPI = exp.path_DAPI
    path_DAPI_out = exp.path_DAPI_out
    path_ChAT = exp.path_ChAT
    path_ChAT_out = exp.path_ChAT_out
    path_out = exp.path_results
    image_format = exp.image_format
    num_images_to_process = exp.num_images_to_process
    out_nucleus_band  = exp.out_nucleus_band
    out_cyto_band = exp.out_cyto_band
    out_inclusion_band = exp.out_inclusion_band
    bwlabel_connection_type = 8
    white_tone = 255;
    black_tone = 0;
    light_gray_tone = 210;
        
    
    #--------------------------------------------------------
    #Creating the output path for the images
    if not os.path.exists(path_out):
        os.makedirs(path_out)
        
        
    #--------------------------------------------------------
    #Getting the names of the images
    str_to_search_DAPI_out = path_DAPI_out + '/*' + image_format
    image_list_DAPI_out = glob.glob(str_to_search_DAPI_out);
    num_images_DAPI_out = len(image_list_DAPI_out);
    if(num_images_to_process != 0):
        image_list_DAPI_out = image_list_DAPI_out[0:num_images_to_process]
        
    str_to_search_ChAT_out = path_ChAT_out + '/*' + image_format
    image_list_ChAT_out = glob.glob(str_to_search_ChAT_out);
    num_images_ChAT_out = len(image_list_ChAT_out);

        
    #--------------------------------------------------------
    #Checking if the number of DAPI and ChAT images match
    print("Number of DAPI Images = %d\nNumber of ChAT Images = %d" % (num_images_DAPI_out, num_images_ChAT_out))
    if (num_images_DAPI_out != num_images_ChAT_out):
        print("ERROR (Overlapping): Number of images in DAPI and ChAT must be the same")
       
    #--------------------------------------------------------
    #Execution flag to see if the module was sucessful
    flag_success = 1

    
    for h in range(num_images_DAPI_out):
        
        #--------------------------------------------------------
        #Reading the Images
        current_image_name = image_list_DAPI_out[h]
        current_image_name = current_image_name[current_image_name.rfind("\\")+1:len(current_image_name)]
        print("Processing Image %d of %d (%s)" % (h+1, num_images_DAPI_out, current_image_name))
                
        #DAPI processed image
        current_DAPI_image = cv2.imread((path_DAPI_out + "/" + current_image_name), cv2.IMREAD_GRAYSCALE)
        img_DAPI = current_DAPI_image
        img_DAPI = (255 - img_DAPI)
        img_DAPI = img_DAPI.astype(numpy.uint8)
        num_rows = img_DAPI.shape[0]
        num_cols = img_DAPI.shape[1]
        
        #ChAT Processed image
        current_ChAT_image = cv2.imread((path_ChAT_out + "/" + current_image_name), cv2.IMREAD_GRAYSCALE)
        img_ChAT = current_ChAT_image
        img_ChAT = img_ChAT.astype(numpy.uint8)
    
        #Original DAPI and ChAT Images
        imgChAT_Original = cv2.imread((path_ChAT + "/" + current_image_name), cv2.IMREAD_GRAYSCALE)
        imgChAT_Original = imgChAT_Original.astype(numpy.uint8)
        imgDAPI_Original = cv2.imread((path_DAPI + "/" + current_image_name), cv2.IMREAD_GRAYSCALE)
        imgDAPI_Original = imgDAPI_Original.astype(numpy.uint8)

        #--------------------------------------------------------
        #Getting the Blobs of the images        
        num_components_ChAT, labels_ChAT = cv2.connectedComponents(img_ChAT, bwlabel_connection_type);
        num_components_ChAT -= 1
        num_components_DAPI, labels_DAPI = cv2.connectedComponents(img_DAPI, bwlabel_connection_type)
        num_components_DAPI -= 1

        #--------------------------------------------------------
        #Verify the size of the bodies and preserve just the image that
        #contains less than a percentage of the image size
        maximum_body_percentage = exp.maximum_body_percentage
        total_number_of_pixels = num_rows*num_cols;
        total_number_of_pixels = float(total_number_of_pixels);
        consider_images = [1, 1]; #[1] = S, [2] = F
               
        #----------------------------------------------------------
        #Analyzing if any of the bodies is too great to analyze
        #I kept this specific calculation but it won't really be used for now
        for i in range(1, num_components_DAPI+1, 1):
            number_of_label_appearance = (labels_DAPI==i).sum()
            current_percentage = number_of_label_appearance / total_number_of_pixels;
            if (current_percentage >=  maximum_body_percentage):
                consider_images[0] = 0;

        for i in range(1, num_components_ChAT+1, 1):
            number_of_label_appearance = (labels_ChAT == i).sum()
            current_percentage = number_of_label_appearance / total_number_of_pixels;
            if (current_percentage >=  maximum_body_percentage):
                consider_images[1] = 0;

        #turning off images
        if(consider_images[0] == 0):
            labels_DAPI = numpy.zeros((num_rows, num_cols), dtype=numpy.uint8);
            

        if(consider_images[1] == 0):
            labels_ChAT = numpy.zeros((num_rows, num_cols), dtype=numpy.uint8);
          

        #----------------------------------------------------------
        #Defining which bodies should be preserved
        label_areas_ChAT = numpy.zeros(num_components_ChAT+1, dtype=numpy.uint32);
        label_areas_DAPI = numpy.zeros(num_components_DAPI+1, dtype=numpy.uint32);
    
        intersection_mapping_ChAT = numpy.zeros(num_components_ChAT+1, dtype=numpy.uint8);
        intersection_mapping_DAPI = numpy.zeros(num_components_DAPI+1, dtype=numpy.uint8);
    
        intersection_mapping_ChAT_only_one_body = numpy.zeros(num_components_ChAT+1, dtype=numpy.uint8);
        intersection_mapping_DAPI_only_one_body = numpy.zeros(num_components_DAPI+1, dtype=numpy.uint8);
    
        for i in range(1,num_components_ChAT+1, 1):
            label_areas_ChAT[i] = (labels_ChAT == i).sum()

    
        for i in range(1, num_components_DAPI+1, 1):
            label_areas_DAPI[i] = (labels_DAPI == i).sum()
    
            
        #----------------------------------------------------
        #veryfing the valid intersections For Two Bodies (Nucleus + inclusion)
        intersection_must_be_greater_than_this = exp.intersection_must_be_greater_than_this
        intersection_must_be_smaller_than_this = exp.intersection_must_be_smaller_than_this #eliminate shoulders
        intersection_must_contain_at_least_this_of_DAPIs_body = exp.intersection_must_contain_at_least_this_of_DAPIs_body; #keeping only inclusions and nuclei
        
        number_of_valid_intersections_necessary = 2; #must have an inclusion and a nucleus
        total_motor_neurons = 0;
        only_one_intersection_motor_neuron = 0;


        for i in range(1, num_components_ChAT+1, 1):
            current_number_os_valid_bodies = 0;

            for j in range(1, num_components_DAPI+1, 1):
                num_pixels_intersection = giara_utils.count_intersections_of_two_labels_on_matrices(labels_ChAT, labels_DAPI, i, j);
                size_of_DAPIs_body = label_areas_DAPI[j];
                size_of_ChATs_body = label_areas_ChAT[i];
                if(((num_pixels_intersection) / (size_of_ChATs_body)) >= (intersection_must_be_greater_than_this)):
                    if(((num_pixels_intersection) / (size_of_DAPIs_body)) >= intersection_must_contain_at_least_this_of_DAPIs_body):
                        if(((size_of_DAPIs_body) / (size_of_ChATs_body)) <= intersection_must_be_smaller_than_this):
                            current_number_os_valid_bodies = current_number_os_valid_bodies + 1;

            #It must have only two: one for the inclusion and one for the
            #nucleus
            if(current_number_os_valid_bodies == number_of_valid_intersections_necessary):
                intersection_mapping_ChAT[i] = i;
                total_motor_neurons = total_motor_neurons + 1;
            elif(current_number_os_valid_bodies == 1):
                only_one_intersection_motor_neuron = only_one_intersection_motor_neuron + 1;
                intersection_mapping_ChAT_only_one_body[i]  = i;

        #----------------------------------------------------
        #now crossing with DAPIs bodies for two intersections
        for i in range(1, num_components_ChAT+1, 1):
            if(intersection_mapping_ChAT[i] != 0):
                for j in range(1, num_components_DAPI+1, 1):
                    num_pixels_intersection = giara_utils.count_intersections_of_two_labels_on_matrices(labels_ChAT, labels_DAPI, i, j);
                    size_of_DAPIs_body = label_areas_DAPI[j];
                    size_of_ChATs_body = label_areas_ChAT[i];

                    #looking if its a valid intersection
                    if(((num_pixels_intersection) / (size_of_ChATs_body)) >= (intersection_must_be_greater_than_this)):
                        if(((num_pixels_intersection) / (size_of_DAPIs_body)) >= intersection_must_contain_at_least_this_of_DAPIs_body):
                            if(((size_of_DAPIs_body) / (size_of_ChATs_body)) <= intersection_must_be_smaller_than_this):
                                intersection_mapping_DAPI[j] = intersection_mapping_ChAT[i];

        #----------------------------------------------------
        #And Now for One Intersection
        for i in range(1, num_components_ChAT+1, 1):
            if(intersection_mapping_ChAT_only_one_body[i] != 0):
                for j in range(1, num_components_DAPI+1, 1):
                    num_pixels_intersection = giara_utils.count_intersections_of_two_labels_on_matrices(labels_ChAT, labels_DAPI, i, j);
                    size_of_DAPIs_body = label_areas_DAPI[j];
                    size_of_ChATs_body = label_areas_ChAT[i];

                    #looking if its a valid intersection
                    if(((num_pixels_intersection) / (size_of_ChATs_body)) >= (intersection_must_be_greater_than_this)):
                        if(((num_pixels_intersection) / (size_of_DAPIs_body)) >= intersection_must_contain_at_least_this_of_DAPIs_body):
                            if(((size_of_DAPIs_body) / (size_of_ChATs_body)) <= intersection_must_be_smaller_than_this):
                                intersection_mapping_DAPI_only_one_body[j] = intersection_mapping_ChAT_only_one_body[i];
                                
        #----------------------------------------------------
        #Now creating the masks
        current_mask_ChAT = numpy.zeros((num_rows, num_cols), dtype=numpy.uint8);
        current_mask_DAPI = numpy.zeros((num_rows, num_cols), dtype=numpy.uint8);


        for i in range(1, num_components_ChAT+1, 1):
            if(intersection_mapping_ChAT[i] != 0):
                current_mask_ChAT = giara_utils.bsxfun("plus", current_mask_ChAT, giara_utils.create_a_mask_for_specific_tone(labels_ChAT, i));
 
                for j in range(1, num_components_DAPI+1, 1):
                    if(intersection_mapping_DAPI[j] == intersection_mapping_ChAT[i]):
                        current_mask_DAPI = giara_utils.bsxfun("plus", current_mask_DAPI, giara_utils.create_a_mask_for_specific_tone(labels_DAPI, j));

        #Now Creating the Masks for only one body
        current_mask_ChAT_only_one_body = numpy.zeros((num_rows, num_cols));
        current_mask_DAPI_only_one_body = numpy.zeros((num_rows, num_cols));

        for i in range(1, num_components_ChAT+1, 1):
            if(intersection_mapping_ChAT_only_one_body[i] != 0):
                current_mask_ChAT_only_one_body = giara_utils.bsxfun("plus", current_mask_ChAT_only_one_body, giara_utils.create_a_mask_for_specific_tone(labels_ChAT, i));
                #current_mask_ChAT_only_one_body = current_mask_ChAT_only_one_body + giara_utils.create_a_mask_for_specific_tone(labels_ChAT, i)

                for j in range(1, num_components_DAPI+1, 1):
                    if(intersection_mapping_DAPI_only_one_body[j] == intersection_mapping_ChAT_only_one_body[i]):
                        current_mask_DAPI_only_one_body = giara_utils.bsxfun("plus", current_mask_DAPI_only_one_body, giara_utils.create_a_mask_for_specific_tone(labels_DAPI, j));
        
        #----------------------------------------------------
        #Now segmenting and preparing the output image
        logical_mask_DAPI = giara_utils.bsxfun("rdivide", current_mask_DAPI, white_tone);
        logical_mask_ChAT = giara_utils.bsxfun("rdivide", current_mask_ChAT, white_tone);
        
        joint_logical = giara_utils.bsxfun("or", logical_mask_ChAT, logical_mask_DAPI);

        logical_mask_DAPI_only_one_body = giara_utils.bsxfun("rdivide", current_mask_DAPI_only_one_body, white_tone);
        logical_mask_ChAT_only_one_body = giara_utils.bsxfun("rdivide", current_mask_ChAT_only_one_body, white_tone);
        joint_logical_only_one_body = giara_utils.bsxfun("or", logical_mask_ChAT_only_one_body, logical_mask_DAPI_only_one_body);
        joint_logical_only_one_body_nuclei_output = giara_utils.bsxfun("and", joint_logical_only_one_body, logical_mask_DAPI_only_one_body);

        joint_logical_only_one_body_cytoplasm_output = giara_utils.bsxfun("and", joint_logical_only_one_body, logical_mask_ChAT_only_one_body);
        img_joint_only_one_body = giara_utils.bsxfun("times", imgChAT_Original, joint_logical_only_one_body);
        img_joint = giara_utils.bsxfun("times", imgChAT_Original, joint_logical);

        #------------------------------------------------------
        #making the color intesity check (Separating Nucleus and Inclusion)
        #col 1 = value, col 2 = index in the component array
        bodies_intensities = numpy.zeros((number_of_valid_intersections_necessary*(total_motor_neurons+1), 2), dtype=numpy.uint8);
        segmented_area_DAPI = giara_utils.bsxfun("times", img_joint, logical_mask_DAPI);
        current_motor_neuron = 1;
        for i in range(1, num_components_ChAT+1, 1):
            if(intersection_mapping_ChAT[i] != 0):

                for j in range(1, num_components_DAPI+1, 1):
                    if(intersection_mapping_DAPI[j] == intersection_mapping_ChAT[i]):
                        if(bodies_intensities[2*current_motor_neuron -1][0] == 0.0):
                            temp = giara_utils.get_mean_intensity_of_label_area(segmented_area_DAPI, labels_DAPI, j);
                            bodies_intensities[2*current_motor_neuron -1][0] = temp;
                            bodies_intensities[2*current_motor_neuron -1][1] = j;
                        else:
                            temp = giara_utils.get_mean_intensity_of_label_area(segmented_area_DAPI, labels_DAPI, j);
                            bodies_intensities[2*current_motor_neuron][0] = temp;
                            bodies_intensities[2*current_motor_neuron][1] = j;

                current_motor_neuron =  current_motor_neuron + 1;
        
        nucleus_inclusion_mask = current_mask_DAPI;
        
        for i in range(1, total_motor_neurons+1, 1):
            darkening_degree = 30;
            darkening_multiplier = 2.0;

            if(bodies_intensities[(2*i - 1)][0] > bodies_intensities[2*i][0]):
                nucleus_inclusion_mask = giara_utils.use_labels_to_change_mask_intensity(nucleus_inclusion_mask, labels_DAPI, bodies_intensities[(2*i - 1)][1], (white_tone - (darkening_degree)));
                nucleus_inclusion_mask = giara_utils.use_labels_to_change_mask_intensity(nucleus_inclusion_mask, labels_DAPI, bodies_intensities[2*i][1], (white_tone - (darkening_degree*darkening_multiplier)));
            else:
                nucleus_inclusion_mask = giara_utils.use_labels_to_change_mask_intensity(nucleus_inclusion_mask, labels_DAPI, bodies_intensities[2*i][1],(white_tone - (darkening_degree)));
                nucleus_inclusion_mask = giara_utils.use_labels_to_change_mask_intensity(nucleus_inclusion_mask, labels_DAPI, bodies_intensities[(2*i - 1)][1], (white_tone - (darkening_degree*darkening_multiplier)));

        inclusions_tone = (white_tone - (darkening_degree))
        nuclei_tone = (white_tone - (darkening_degree*darkening_multiplier))


        #-----------------------------------------------------------------
        #Eliminating only inclusions for single body
        joint_post_brightness_processing = img_joint_only_one_body[:,:];
        num_joint_components, joint_components_labels = cv2.connectedComponents(joint_post_brightness_processing, bwlabel_connection_type);
        num_joint_components -= 1
        maximum_motor_neuron_intensity = 51.0;


        for i in range(1, num_joint_components+1, 1):
            current_intensity = giara_utils.get_mean_intensity_of_label_area(joint_post_brightness_processing, joint_components_labels, i);
            if(current_intensity  > maximum_motor_neuron_intensity):
                joint_post_brightness_processing = giara_utils.use_labels_to_change_mask_intensity(joint_post_brightness_processing, joint_components_labels, i, 0);
                only_one_intersection_motor_neuron = only_one_intersection_motor_neuron - 1;

        #-----------------------------------------------------------------
        #Treating the colors of the output images and separating them
        joint_post_brightness_processing_temp = numpy.zeros((num_rows, num_cols, 3));
        joint_post_brightness_processing_temp[:,:,0] = joint_post_brightness_processing;
        joint_post_brightness_processing_temp[:,:,1] = joint_post_brightness_processing;
        joint_post_brightness_processing_temp[:,:,2] = joint_post_brightness_processing;
        joint_post_brightness_processing = joint_post_brightness_processing_temp;
        joint_two_bodies_with_one_body_after_brightness_evaluation = magnolia_utils.imjoin_two_bodies_with_one_body_3d(giara_utils.gs_to_rgb(img_joint), joint_post_brightness_processing, black_tone);

        #Joining the cytoplasm and nuclei of the nuerons with only nuclei
        #with the neurons with nuclei and inclusions, and separating each output class (Nuclei, Inclusions and Cytoplasm)
        cyto_output = magnolia_utils.img_cytoplasm_creation(joint_two_bodies_with_one_body_after_brightness_evaluation, [0, 0, 0], exp.out_cyto_band);
        nuclei_output, inclusions = magnolia_utils.img_nuclei_and_inclusions_creation(nucleus_inclusion_mask, nuclei_tone, inclusions_tone, exp.out_nucleus_band, exp.out_inclusion_band);
        cyto_output = magnolia_utils.img_cytoplasm_refinement(cyto_output, inclusions, nuclei_output, exp.out_nucleus_band, exp.out_inclusion_band);
        cyto_output = magnolia_utils.cyto_logical_add_one_body(cyto_output, joint_logical_only_one_body, exp.out_cyto_band)
        nuclei_only_one_body = current_mask_DAPI_only_one_body
        nuclei_output = magnolia_utils.add_nuclei_from_only_one_body(nuclei_output, nuclei_only_one_body, exp.out_nucleus_band);
        temp_term_1 = giara_utils.bsxfun("rdivide", joint_post_brightness_processing,  white_tone)
        temp_term_2 = giara_utils.bsxfun("times", giara_utils.bsxfun("rdivide", nuclei_only_one_body, white_tone), (-1))
        joint_logical_only_one_cyto = giara_utils.bsxfun("plus", temp_term_1, giara_utils.gs_to_rgb(temp_term_2));
        joint_logical_only_one_cyto = giara_utils.bsxfun("times", joint_logical_only_one_cyto, white_tone);
        cyto_output = magnolia_utils.add_cytoplasm_from_only_one_body_nuclei(cyto_output, nuclei_output, exp.out_nucleus_band);

        #-----------------------------------------------------------------
        #Writing the four output images
        cv2.imwrite((path_out + "/" + current_image_name.replace(".png", "_Cytoplasms.tiff")), cyto_output);
        cv2.imwrite((path_out + "/" + current_image_name.replace(".png", "_Inclusions.tiff")), inclusions);
        cv2.imwrite((path_out + "/" + current_image_name.replace(".png", "_Nuclei.tiff")), nuclei_output);
        cv2.imwrite((path_out + "/" + current_image_name.replace(".png", "_All.tiff")), magnolia_utils.img_join_cyto_nu_and_inc(cyto_output, nuclei_output, inclusions, exp.out_nucleus_band, exp.out_inclusion_band));
        
                    
    return flag_success;
