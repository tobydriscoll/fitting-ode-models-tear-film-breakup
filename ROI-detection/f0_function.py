import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os, glob
from PIL import Image
import cv2
from scipy import signal
from scipy.optimize import minimize, root
import TBU_Functions as tbu

## FUNCTIONS!

def get_im(image_path, cor_doc, single_image = False,v = 4):
    # Use the cornea document to find the lower meniscus
    if(v == 4):
        x_y_r = np.mean(np.array(pd.read_csv(cor_doc,names = None, header = 0)[['cenrow', 'cencol', 'radius']]),axis = 0)
        print(x_y_r)
    elif(v==3):
        x_y_r = np.mean(np.array(pd.read_csv(cor_doc,names = None)),axis= 0)
    
    if(single_image == False):
        os.chdir(image_path)
        files = sorted(glob.glob("*.png"))
        ref_image = files[-3]
    else:
        ref_image = image_path
        
    img = Image.open(ref_image)
    w, h = img.size
    pm = h/8 # plus/minus to create box of height h/4
    tm_box_w = 100 # narrow width of box to look at meniscus (from Ziwei)

    #x_y_r = np.mean(x_y_r[0:len(files)], axis = 0)
    
    meniscus = [h*x_y_r[1], h*(x_y_r[0] + x_y_r[2])]
    
    box_cord = [int((int(meniscus[1]) - pm)), int((int(meniscus[1]) + pm)), \
             int((int(meniscus[0]) - tm_box_w/2)), int((int(meniscus[0]) + tm_box_w/2))]
    box_cord[1] = min(box_cord[1], np.shape(img)[0])
    
    height = box_cord[1] - box_cord[0]
    
    return height, box_cord

def image_curve( I_crop, box_cord):
# this program is designed to draw the curve for image
# INPUT:
# I_crop is the gray cropped image
# box_cord are the coordinates of the cropped box
# OUTPUT:
# x is the number sequence of the y axis of image
# x_mean is the mean FL value along the vertical line
# x_std is the std FL value along the vertical line

# use autocorrelation to move the figure
# all columns are compared to 1st
    y1 = I_crop[:, 0]
    
    width = box_cord[3] - box_cord[2]
    height = box_cord[1] - box_cord[0]
    
    index = np.zeros((width,1))
    step = np.zeros((width,1))
    for ii in range(width):
        c = signal.correlate(y1.astype('float'), (I_crop[:,ii]).astype('float'), "full")
        lags = signal.correlation_lags(y1.size, I_crop[:,ii].size)
        index[ii] = np.argmax(c)
        step[ii] = lags[int(index[ii])]

    blank = min(step)

# average the data into one data set
    row = round(height*2)
    column = round(width)
    final = np.ones((row, column))*400

    for j in range(width): 
        zw1 = round(height)
        zw2 = range(1, zw1+1)
        x_value = zw2+ step[j]+ np.abs(blank)
        srt = int(x_value[0])
        fnsh = int(x_value[-1])
        final[srt:(fnsh+1), j]= I_crop[:, j]       

    x_mean = np.zeros((row,1))
    for i in range(row):
        ind = final[i, :] < 400
        x_final = final[i, ind]
        x_mean[i] = np.mean(x_final)

    return x_mean

def autocorr_frames(a1, b1):
    
    # Function to autocorrelate the frames
    #Find non-NaN values
    ind1 = np.argwhere(np.isnan(a1) == False)
    ind2 = np.argwhere(np.isnan(b1) == False)

    a = a1[ind1] # pick out the non-NaN values
    b = b1[ind2] # pick out the non-Nan values

    # do cross correlation
    c = signal.correlate(a.astype('float'), b.astype('float'), "full")
    lags = signal.correlation_lags(a.size, b.size) # find the lags
    index = np.argmax(c) # find the index of the max element
    step = lags[int(index)] # find the lag of the max element

    return step

def step2image(raw_image, step):

# create a new aligned image after autocorrelation of the raw image
# step is the matrix

    blank1 = np.abs(min(step)) + 5
    blank2 = max(step) + 5
    a, b = np.shape(raw_image)

    aligned_image = np.zeros((int(blank1) + int(blank2) + a, b))
    for i in range(b):
        vertical_position = np.array(range(a)) + int(step[i]) + blank1
        aligned_image[vertical_position.astype(int), i] = raw_image[:, i]

    return aligned_image

def  estimate_FL_peak_v2( h1 ):
# estimate the maximum FL intensity and corresponding FL concentration 
# with the theoretical formula I = (1 -exp(-kappa*h*f))/(1 + (f/fcr)^2)
# input:
#       h1: estimated tear film thickness in micrometer
# ouput: 
#       I_max: maximum FL intenssity
#       f_peak: FL concentration which gives the maximum FL intensity

# set up the parameters 
    fcr = 0.2 # critical FL concentration 0.2%
    WaterDensity = 1000 
    FLMolWt = 376 # molecular weight of FL g/M
    ToMassFrac = 1e2
    ConvFactor = WaterDensity/FLMolWt/ToMassFrac;  
    kappa = 1.75e7  # Naperian, base e, per meter, per molar
    kappa = kappa*ConvFactor

    h = h1*10**(-6) # tear film thickness (from micrometer to m)

# setup the formula -I = -(1 -exp(-kappa*h*f))/(1 + (f/fcr)^2)
    f2i = lambda f: - (1-np.exp(-kappa*h*f))/(1 + (f/fcr)**2)
# search for the minimum of f21, which is the maximum of I
    sol = minimize(f2i, np.array([1]))
    f_peak = sol.x

    return f_peak

def model_i2f_rjb(I_ratio, f_peak, h1):
# this program is to calculate the FL concentration given an intensity
# Matlab function fzero is used to find the two concentrations values

# Input: 
#     I_ratio: the ratio FL intensity
#     f_peak: the FL concentration with highest FL intensity
#     h1: the thickness of the film (unit:micron)
# Output: 
#     sol1: low FL concentration
#     sol2: high FL concentration

# parameters for the equation
    f0 = 0.2         # critical FL concentration 0.2%
    a = 1.75e5       # unit: cm^(-1) M^(-1)  (Naperian value, i.e., base e)
    h = h1*10**(-4)   # tear film thickness (5 micrometer, m)
    weight = 376     # molecular weight of FL g/mole

# equation to be solved for fluorescein concentration (called x1 here)
# note that 10/weight converts a to units of cm^(-1) %^(-1)
    equa1 = lambda x1: I_ratio - (1 - np.exp(-(a*10/weight)*x1*h))/(1+(x1/f0)**2)

# solve for the left (low) side first
# set interval for answer to be on left of f_peak

    sol = root(equa1, np.array([f_peak/2]))
    sol1 = sol.x
    
# now solve for the right (high) side
# set interval for answer to be to the right of f_peak

    sol = root(equa1,np.array([2*f_peak]))
    sol2 = sol.x
    
    return sol1, sol2 

def f0_estimate(image_path, cor_doc, single_image = True,v=4):
    
    # Put everything together into a single function to estimate the 
    # initial fluorescein concentration (f_0)
    # Input: path to folder of dark images (no blinks, close to first bright image if possible)
    # Input: cornea document that contains the information needed to find the lower meniscus
    # Input: option to tell the code if the folder contains only one image (should use multiple)
    # Output: f_0 value given as a percentage. 0.2 means 0.2% (critical concentration)
    
    conv2mm = 0.0058 # conversion factor from number of pixels to mm
    THR_BL = 0.5 # threshold for tear meniscus height
    Ical = 228.67  # Debbie's data, low intensity light source
    
    l_d,img_num = get_ld(image_path) #use the last_dark image
    height, box_cord = get_im(l_d,cor_doc,single_image = single_image, v = v)

    #os.chdir(image_path)
    unused_path = image_path[0:int(len(image_path)-5)] + '_unused' + os.sep + '*.png'  
    files = sorted(glob.glob(unused_path))[img_num-6:img_num-1]  #extracts last 5 images
    x_mean = np.zeros((len(files),2*height,1))

    for i in range(len(files)):
        I = cv2.imread(files[i])
        I = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)
        I_crop = I[box_cord[0]:box_cord[1], box_cord[2]:box_cord[3]]
        x_mean[i,:] = image_curve(I_crop, box_cord)

    x_mean = x_mean[:,0:height,0]
    x_mean = np.transpose(x_mean)

    step = np.zeros((len(files)))
    for i in range(len(files)):
        step[i] = autocorr_frames(x_mean[:,0], x_mean[:,i])

    men_image = step2image(x_mean, step)
    men_image[np.argwhere(np.isnan(men_image) == True)] = 0 # set NaNs to zero

    thresh = THR_BL*np.max(men_image) # threshold to find the meniscus (50%)
    men_loc = np.where(men_image > thresh)
    top = min(men_loc[0])
    bottom = max(men_loc[0])

    TMHcutoff = np.zeros((len(files)))   
    top_halfLoc = []
    bottom_halfLoc = []
    TMH = np.zeros((len(files)))
    I_max1 = np.zeros((len(files)))

    for i in range(len(files)):
        Max_data = np.argmax(men_image[top:bottom, i])
        Max_Loc = Max_data + top
        TMHcutoff[i] = max(men_image[top:bottom, i])*THR_BL   #set as percentage of max (50%) 
        subtr = np.abs(men_image[:,i] - TMHcutoff[i])

        top_halfLoca = np.where(subtr[range(Max_Loc)] == min(subtr[range(Max_Loc)]))
        top_halfLoc = np.append(top_halfLoc,top_halfLoca[0][0])

        if (top_halfLoc[i].size == 0):
            top_halfLoc[0][i] = 1

        bottom_halfLoca = np.where(subtr[Max_Loc:-1] == min(subtr[Max_Loc:-1])) + (Max_Loc - 1)  
        bottom_halfLoc = np.append(bottom_halfLoc, bottom_halfLoca[0][0])

        if (bottom_halfLoc[i].size == 0):
            bottom_halfLoc[i] = 2

        TMH[i] = bottom_halfLoc[i] - top_halfLoc[i]

        I_max1[i] = np.max(men_image[top_halfLoc[0].astype(int):bottom_halfLoc[-1].astype(int),:])                                                      

    TMH_mm = np.mean(TMH)*conv2mm # unit: mm 
    TM_deep = 1000*TMH_mm/1.56 # convert to microns; literature conversion rate

    f_peak  = estimate_FL_peak_v2(TM_deep)

    I_ratio = np.zeros((len(files)))
    sol1 = np.zeros((len(files)))
    sol2 = np.zeros((len(files)))

    for i in range(len(files)):
        I_ratio[i] = I_max1[i]/Ical
        sol1[i], sol2[i] = model_i2f_rjb(I_ratio[i], f_peak, TM_deep)

    sol_ops = np.array([np.mean(sol1), np.mean(sol2)])

    f_0_est = (sol_ops[(sol_ops > 1e-2)])[0] # ignore the tiny FL concentration option
    
    return f_0_est

def get_ld(image_path):
    files = sorted(glob.glob(image_path))
    f_b = files[0]   # first bright image
    
    if(int(f_b.split(os.sep)[-2][1:]) == 10): #check if on trial t10 vs t1-t9 extra character  
        end =  22
    else:
        end = 21
    
    img_num_f_b = int(f_b.split(os.sep)[-1].split("_")[-1].split(".")[0]) #extracts image number
    previous_img = tbu.get_padding(img_num_f_b-1) + str(img_num_f_b-1)+ '.png'      #creates file name for last dark image
    l_d = f_b[0:len(f_b)-end] + '_unused' + os.sep + f_b[len(f_b)-end:len(f_b)-8] + previous_img #makes file_path for l_d
    return l_d,img_num_f_b