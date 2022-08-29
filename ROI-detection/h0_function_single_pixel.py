import numpy as np
import warnings
warnings.filterwarnings("ignore")
import TBU_Functions as tbu
import glob
import os
from tensorflow.keras.preprocessing import image
import cv2

def get_h0(image_path,cor_doc,boxes,locations,f_0,frame_rate,tile_size = 7,sigma = 10,box_size = 192,v=3):
    #3x3 
    #intilzation / Parameters:
    #adv_x_y = tbu.get_adv_x_y_skip(image_path,cor_doc,boxes,frame_rate,3,method ='median')
    fcr = 0.0053  # critical concentration in molar (M)
    epsf = 1.75e7 # in m^-1 M^-1
    d_char = 3.5e-6    # in m
    phi = epsf*d_char*fcr   # exponent parameter
    f = np.logspace(-2,1,501)
    kernel = int(np.ceil(4*sigma) + 1)
    
    files = sorted(glob.glob(image_path))
    I_0_est = 104.02900089347337  #I_0 by fitting model eye data
    
    #path to:
    f_b = files[0]   # first bright image
    l_b = files[-1]  # last bright image
    
    if(int(f_b.split(os.sep)[-2][1:]) == 10): #check if on trial t10 vs t1-t9 extra character  
        end =  22
    else:
        end = 21
    
    #swichted to 2 images before    
    img_num_f_b = int(f_b.split(os.sep)[-1].split("_")[-1].split(".")[0]) #extracts image number
    previous_img = tbu.get_padding(img_num_f_b-2) + str(img_num_f_b-2)+ '.png'      #creates file name for last dark image
    l_d = f_b[0:len(f_b)-end] + '_unused' + os.sep + f_b[len(f_b)-end:len(f_b)-8] + previous_img #makes file_path for l_d
    
    h_0 = np.zeros(len(boxes)) #create h_0 array to store each value
    minI = np.zeros(len(boxes))+ 1000 # initialize as something large need to keep minI values also
    
    for i in range(len(boxes)): 
        loc_min = np.zeros((len(files),1)) # initialize loc min array #also re-set array every loop
        for j in range(len(files)):
            ref_image = tbu.get_crop(files[j],cor_doc,single_image = True,show_image = False,v=v)
             #box location realitve to cropped cornea image
            grey_image = cv2.cvtColor(image.img_to_array(ref_image.crop(boxes[i]).resize((96,96))),cv2.COLOR_BGR2GRAY) #res
            #grey_image = cv2.cvtColor(image.img_to_array(ref_image.crop(boxes[i])),cv2.COLOR_BGR2GRAY)
            blured_image = cv2.GaussianBlur(grey_image,(kernel,kernel),sigma,cv2.BORDER_REFLECT)
            #r,c = np.where(blured_image == blured_image.min())
            #loc_min[j] = np.mean(blured_image[int(np.mean(r)):(int(np.mean(r)) + tile_size), int(np.mean(c)):(int(np.mean(c))+tile_size)])
            center_r = int(np.median(locations[i][:,0]))
            center_c = int(np.median(locations[i][:,1]))
            #center_r = int(locations[i,0])
            #center_c = int(locations[i,1])
            loc_min[j] = blured_image[center_r,center_c]
            if loc_min[j] < minI[i]:
                minI[i] = loc_min[j]
    

    #new box size
    siz = tile_size*tile_size # change to alter size of (square) box to average
    a_s = round((siz-1)/2)
    
    #read in frist bright and last dark images
    f_b_2 = files[2]
    bright_im = tbu.get_crop(f_b_2,cor_doc,single_image = True,show_image = False,v=v)
    dark_im = tbu.get_crop(l_d,cor_doc,single_image = True,show_image = False,v=v)
    
    #get adv in new box size on f_b and l_d images
    center = int((box_size/2)) #square box  x and y always gonna be the same number for center
    
    left = int(((tile_size-1)/2))
    right = int(((tile_size-1)/2) + 1)
    
    for k in range(len(boxes)):
        
       
        #do mods to first bright
        bright_grey_image = cv2.cvtColor(image.img_to_array(bright_im.crop(boxes[k]).resize((96,96))),cv2.COLOR_BGR2GRAY)
        bright_blured_image = cv2.GaussianBlur(bright_grey_image,(kernel,kernel),sigma,cv2.BORDER_REFLECT)
        
        #do mods to last dark
        dark_grey_image = cv2.cvtColor(image.img_to_array(dark_im.crop(boxes[k]).resize((96,96))),cv2.COLOR_BGR2GRAY)
        dark_blured_image = cv2.GaussianBlur(dark_grey_image,(kernel,kernel),sigma,cv2.BORDER_REFLECT)
    
        #sq_2_avg_f_b = np.mean(bright_blured_image[(center-a_s):(center+a_s+1), (center-a_s):(center+a_s+1)])
        #sq_2_avg_l_d = np.mean(dark_blured_image[(center-a_s):(center+a_s+1), (center-a_s):(center+a_s+1)])
        
        
        center_r = int(np.median(locations[k][:,0]))
        center_c = int(np.median(locations[k][:,1]))
        
        sq_2_avg_f_b = np.mean(bright_blured_image[center_r-left:center_r+right,center_c-left:center_c+right])
        sq_2_avg_l_d = np.mean(dark_blured_image[center_r-left:center_r+right,center_c-left:center_c+right])
        
    
        if(np.isnan(sq_2_avg_f_b)):
            #check row left
            if(center_r-left < 0):
                r_l = 0
            else:
                r_l = center_r - left
            
            #check row right
            if(center_r+right > 96 ):
                r_r = 96
            else:
                r_r = center_r + right
            
            #check col left 
            if(center_c-left < 0):
                c_l = 0
            else:
                c_l = center_c - left
            
            #check col right
            if(center_c+ right > 96):
                c_r = 96
            else:
                c_r = center_c + right
        
            
            sq_2_avg_f_b = np.mean(bright_blured_image[r_l:r_r,c_l:c_r])
            sq_2_avg_l_d = np.mean(dark_blured_image[r_l:r_r,c_l:c_r])

            
    
        #print(sq_2_avg_f_b,sq_2_avg_l_d,center_r,center_c)
        ld_scale = (sq_2_avg_f_b/sq_2_avg_l_d)
        #Solve for H0
        I_square = sq_2_avg_f_b - minI[k]

        findh_0 = lambda phi, f, I, I_0: -(1/(phi*f))*np.log(1-(I/I_0)*(1+f**2))
        fcrit = 0.2
        f_nd = f_0/fcrit
        I_0_scaled = I_0_est*ld_scale
 
        # find h as a fraction of d
        h_0_value = findh_0(phi, f_nd, I_square, I_0_scaled)
        #print(ld_scale,(1-(I_square/I_0_scaled)*(1+f_nd**2)))
        h_0[k] = h_0_value*d_char*1e6 # h_0 in micrometers
        drep = d_char*1e6
        #print("Using d = ", drep, " micrometers, the initial thickness estimate is ", h_0_dim, " micrometers.")
        #print("The minimum intensity value to subtract off in subsequent codes is ", minI, ".")
    new_h0 = np.reshape(h_0,(1,len(boxes)))    #change both to row vectors and append together
    new_min = np.reshape(minI,(1,len(boxes)))
    h0_min = np.append(new_h0,new_min,0)
    
    return h0_min