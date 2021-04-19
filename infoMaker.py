import os as os

class Experiment:
    #-----------------------------------------------
    #General Vars
    #-----------------------------------------------  
    #Root Path of the project
    path_project = "C:/Users/felipe.antunes/OneDrive - UPNA/00_-_Projects/001_-_Motor Neurons/Ground_Truth_Selection"
    
    #Output path for the results
    path_output_folder = "/Transcripts"
    
    #Paths of the original images
    path_DAPI = path_project + "/DAPI_Images" 
    path_ChAT = path_project + "/ChAT_Images"
    
    #Paths of the results
    path_DAPI_out = path_project + path_output_folder +"/DAPI_Out"
    path_ChAT_out = path_project + path_output_folder +"/ChAT_Out"
    path_results =  path_project + path_output_folder + "/Overlapping_Out"
    
    #Input Image format
    image_format = "png"
    
    #Number of images to be processed    
    num_images_to_process = 4
        
    #-----------------------------------------------
    #DAPI
    #-----------------------------------------------
    DAPI_min_body_size = 500
    DAPI_number_of_clusters = 4
        
    #-----------------------------------------------
    #ChAT
    #-----------------------------------------------
    ChAT_min_body_size = 1300    
    
    #-----------------------------------------------
    #Overlapping
    #----------------------------------------------
    #Colors for the output images (RGB)
    out_nucleus_band  = [102, 0, 204]
    out_cyto_band = [255, 153, 0]
    out_inclusion_band = [0, 153, 153]

    #Maximum of the image size that a ChAT or DAPI blob can be, in order to be analyzed 
    maximum_body_percentage = 0.10;   
    
    #restraints for the overllaping operation
    intersection_must_be_greater_than_this = 0.05
    intersection_must_be_smaller_than_this = 0.50
    intersection_must_contain_at_least_this_of_DAPIs_body = 0.9




    #-----------------------------------------------
    def __init__(self):
        temp_str = 'Constructor'
        
    def __str__(self):
        temp_str = 'Experiment Vars'
        return temp_str