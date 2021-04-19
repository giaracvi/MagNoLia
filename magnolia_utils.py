import numpy as numpy
import time as time

pure_white = 255

#----------------------------------------------------------------------
#Functions
#---------------------------------------------------------------------- 
def separate_cytoplasm_from_nuclei_for_only_one_body(cyto_output, joint_logical_only_one_body_cytoplasm_output, nuclei_only_one_body, out_cyto_band):
    time_ini = time.time()
    ncols = cyto_output.shape[0];
    nrows = cyto_output.shape[1];
    
    #[nrows, ncols, ~] = size(cyto_output);
    for i in range(nrows):
        for j in range(ncols):
            if(nuclei_only_one_body[i, j] == pure_white):
                joint_logical_only_one_body_cytoplasm_output[i, j] = 0;
    
    for i in range(nrows):
        for j in range(ncols):
            if(joint_logical_only_one_body_cytoplasm_output[i, j] == pure_white):
                cyto_output[i, j, 0] = out_cyto_band[2];
                cyto_output[i, j, 1] = out_cyto_band[1];
                cyto_output[i, j, 2] = out_cyto_band[0];

    str_out = "(MagNoLia Utils) separate_cytoplasm_from_nuclei_for_only_one_body completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return cyto_output


def add_cytoplasm_from_only_one_body_nuclei(cyto_output, nuclei, out_nucleus_band):
    time_ini = time.time()
    ncols = cyto_output.shape[0]
    nrows = cyto_output.shape[1]
    
    #[nrows, ncols, ~] = size(cyto_ouput);
    for i in range(nrows):
        for j in range(ncols):
            if((nuclei[i, j, 0] == out_nucleus_band[2]) and (nuclei[i, j, 1] == out_nucleus_band[1]) and (nuclei[i, j, 2] == out_nucleus_band[0])):
                cyto_output[i, j, 0] = pure_white;
                cyto_output[i, j, 1] = pure_white;
                cyto_output[i, j, 2] = pure_white;

    str_out = "(MagNoLia Utils) add_cytoplasm_from_only_one_body_nuclei completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return cyto_output



def add_cytoplasm_from_only_one_body(cyto_output, cyto_only_one_body, out_cyto_band):
    time_ini = time.time()
    ncols = cyto_output.shape[0]
    nrows = cyto_output.shape[1]
    
    for i in range(nrows):
        for j in range(ncols):
            if(cyto_only_one_body[i, j] == pure_white):
                cyto_output[i, j, 0] = out_cyto_band[2];
                cyto_output[i, j, 1] = out_cyto_band[1];
                cyto_output[i, j, 2] = out_cyto_band[0];
                
    str_out = "(MagNoLia Utils) add_cytoplasm_from_only_one_body completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return cyto_output



def add_nuclei_from_only_one_body(nuclei_output, nuclei_only_one_body, out_nucleus_band):
    time_ini = time.time()
    ncols = nuclei_output.shape[0]
    nrows = nuclei_output.shape[1]
    #[nrows, ncols, ~] = size(nulcei_ouput);
    for i in range(nrows):
        for j in range(ncols):
            if(nuclei_only_one_body[i, j] == pure_white):
                nuclei_output[i, j, 0] = out_nucleus_band[2];
                nuclei_output[i, j, 1] = out_nucleus_band[1];
                nuclei_output[i, j, 2] = out_nucleus_band[0];
                
    str_out = "(MagNoLia Utils) add_nuclei_from_only_one_body completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return nuclei_output


def img_nuclei_and_inclusions_creation(DAPI_img, tone_nu, tone_inc, out_nucleus_band, out_inclusion_band):
    time_ini = time.time()
    ncols = DAPI_img.shape[0]
    nrows = DAPI_img.shape[1]
    #[nrows, ncols, ~] = size(DAPI_img);
    
    nuclei = numpy.zeros((nrows, ncols, 3), dtype=numpy.uint8);
    inclusions = numpy.zeros((nrows, ncols, 3), dtype=numpy.uint8)
    
    for i in range(nrows):
        for j in range(ncols):
            if(DAPI_img[i][j] == tone_nu):
                nuclei[i][j][0] = out_nucleus_band[2];
                nuclei[i][j][1] = out_nucleus_band[1];
                nuclei[i][j][2] = out_nucleus_band[0];
            else:
                nuclei[i][j][0] = pure_white;
                nuclei[i][j][1] = pure_white;
                nuclei[i][j][2] = pure_white;
                
            
            if(DAPI_img[i][j] == tone_inc):
                inclusions[i][j][0] = out_inclusion_band[2];
                inclusions[i][j][1] = out_inclusion_band[1];
                inclusions[i][j][2] = out_inclusion_band[0];
            else:
                inclusions[i][j][0] = pure_white;
                inclusions[i][j][1] = pure_white;
                inclusions[i][j][2] = pure_white;
                
    str_out = "(MagNoLia Utils) img_nulcei_and_inclusions_creation completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return nuclei, inclusions



def img_join_cyto_nu_and_inc(cytoplasm, nuclei, inclusions, out_nucleus_band, out_inclusion_band):
    time_ini = time.time()
    ncols = cytoplasm.shape[0]
    nrows = cytoplasm.shape[1]
    
    #[nrows, ncols, ~] = size(cytoplasm);
    for i in range(nrows):
        for j in range(ncols):
            if((inclusions[i, j, 0] == out_inclusion_band[2]) and (inclusions[i, j, 1] == out_inclusion_band[1]) and (inclusions[i, j, 2] == out_inclusion_band[0])):
                cytoplasm[i, j, 0] = out_inclusion_band[2]
                cytoplasm[i, j, 1] = out_inclusion_band[1]
                cytoplasm[i, j, 2] = out_inclusion_band[0]
                
            if((nuclei[i, j, 0] == out_nucleus_band[2]) and (nuclei[i, j, 1] == out_nucleus_band[1]) and (nuclei[i, j, 2] == out_nucleus_band[0])):
                cytoplasm[i, j, 0] = out_nucleus_band[2]
                cytoplasm[i, j, 1] = out_nucleus_band[1]
                cytoplasm[i, j, 2] = out_nucleus_band[0]

    str_out = "(MagNoLia Utils) img_join_cyto_nu_and_inc completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return cytoplasm



def img_cytoplasm_refinement(cytoplasm, inclusions, nuclei, out_nucleus_band, out_inclusion_band):
    time_ini = time.time()
    ncols = cytoplasm.shape[0]
    nrows = cytoplasm.shape[1]
    #[nrows, ncols, ~] = size(cytoplasm);
    
    for i in range(nrows):
        for j in range(ncols):
            if((inclusions[i, j, 0] == out_inclusion_band[2]) and (inclusions[i, j, 1] == out_inclusion_band[1]) and (inclusions[i, j, 2] == out_inclusion_band[0])):
                cytoplasm[i, j, 0] = pure_white
                cytoplasm[i, j, 1] = pure_white
                cytoplasm[i, j, 2] = pure_white

            if((nuclei[i, j, 0] == out_nucleus_band[2]) and (nuclei[i, j, 1] == out_nucleus_band[1]) and (nuclei[i, j, 2] == out_nucleus_band[0])):
                cytoplasm[i, j, 0] = pure_white;
                cytoplasm[i, j, 1] = pure_white;
                cytoplasm[i, j, 2] = pure_white;

    str_out = "(MagNoLia Utils) img_cytoplasm_refinement completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return cytoplasm




def img_cytoplasm_creation(img_orig, tone_bg, out_cyto_band):
    time_ini = time.time()
    ncols = img_orig.shape[0]
    nrows = img_orig.shape[1]
    #[nrows, ncols, ~] = size(img_orig);
    
    img_out = img_orig;
    
    for i in range(nrows):
        for j in range(ncols):
            if((img_out[i, j, 0] == tone_bg[2]) and (img_out[i, j, 1] == tone_bg[1]) and (img_out[i, j, 2] == tone_bg[0])):
                img_out[i, j, 0] = pure_white;
                img_out[i, j, 1] = pure_white;
                img_out[i, j, 2] = pure_white;
            else:
                img_out[i, j, 0] = out_cyto_band[2];
                img_out[i, j, 1] = out_cyto_band[1];
                img_out[i, j, 2] = out_cyto_band[0];
                
    str_out = "(MagNoLia Utils) img_cytoplasm_creation completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return img_out

def imjoin_two_bodies_with_one_body_3d(img1, img2, tone):
    time_ini = time.time()
    ncols = img1.shape[0]
    nrows = img1.shape[1]
    
    #[nrows, ncols, ~] = size(img1);
    result = numpy.zeros((nrows, ncols, 3), dtype=numpy.uint8);
    
    
    for i in range(nrows):
        for j in range(ncols):
            if ((img1[i, j, 0] == tone) and (img1[i, j, 1] == tone) and (img1[i, j, 2] == tone)):
                if ((img2[i, j, 0] == tone) and (img2[i, j, 1] == tone) and (img2[i, j, 2] == tone)):
                    result[i, j, 0] = tone;
                    result[i, j, 1] = tone;
                    result[i, j, 2] = tone;
                else:
                    result[i, j, 0] = img2[i, j, 0];
                    result[i, j, 1] = img2[i, j, 1];
                    result[i, j, 2] = img2[i, j, 2];
            else:
                result[i, j, 0] = img1[i, j, 0];
                result[i, j, 1] = img1[i, j, 1];
                result[i, j, 2] = img1[i, j, 2];
             
    str_out = "(MagNoLia Utils) imjoin_two_bodies_with_one_body_3d completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return result

def imjoin_two_bodies_with_one_body(img1, img2, tone):
    time_ini = time.time()
    ncols = img1.shape[0]
    nrows = img1.shape[1]

    
    #[nrows, ncols, ~] = size(img1);
    result = numpy.zeros((nrows, ncols), dtype=numpy.uint8);
    
    for i in range(nrows):
        for j in range(ncols):
            if (img1[i][j] == tone):
                if (img2[i][j] == tone):
                    result[i][j] = tone;
                else:
                    result[i][j] = img2[i][j];
            else:
                result[i][j] = img1[i][j];
             
    str_out = "(MagNoLia Utils) imjoin_two_bodies_with_one_body completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return result

def cyto_logical_add_one_body(cyto, logical_mask, out_cyto_band):
    time_ini = time.time()
    ncols = cyto.shape[0]
    nrows = cyto.shape[1]
    
    for i in range(0, nrows, 1):
        for j in range(0, ncols, 1):
            if(logical_mask[i][j] == 1):
                cyto[i][j][0] = out_cyto_band[2]
                cyto[i][j][1] = out_cyto_band[1]
                cyto[i][j][2] = out_cyto_band[0]
                
    str_out = "(MagNoLia Utils) cyto_logical_add_one_body completed (Time Taken: %.2fs)" % (time.time() - time_ini)
    print(str_out)
    return cyto