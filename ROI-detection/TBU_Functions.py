# Created by Dominick Sinopoli  January 2021
# Part_2 of Functions Created for TBU focus extracting data

import glob
import os
from PIL import Image
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import matplotlib.patches as patches
import scipy.cluster.hierarchy as hcluster
import cv2

# Common input parameters for all functions
#    image_path - path to root of trial images (stabilized)
#    model_path - path to trained CNN model
#    cor_doc - path to cornea detection document
#    frame_rate - frame rate of the trial video (either 4 or 5 per second)
#    boxes - list of clipping boxes that designate regions of interest
#    thresh - decision threshold for TBU detection by the CNN

###############################################################################
# Image/file handling
###############################################################################

# Determine a crop rectangle close to the cornea for a trial
# Returns either the crop rectangle or a cropped image
def get_crop(image_path, cor_doc, single_image=False, show_image=True, get_box=False, v=3):
    if (v == 4):
        x_y_r = np.mean(np.array(pd.read_csv(cor_doc, names=None, header=0)[
                        ['cenrow', 'cencol', 'radius']]), axis=0)
    elif (v == 3):
        x_y_r = np.mean(np.array(pd.read_csv(cor_doc, names=None)), axis=0)
    #x_y_r = np.mean(np.array(pd.read_csv(cor_doc,names = None)),axis =0)
    if (single_image == False):
        files = sorted(glob.glob(image_path))
        ref_image = files[-3]
    else:
        ref_image = image_path

    img = Image.open(ref_image)
    width, height = img.size
    right = (x_y_r[1]*height+x_y_r[2]*height)*1.05
    left = (x_y_r[1]*height-x_y_r[2]*height)*.95
    # upper = (x_y_r[0]*height-(x_y_r[2]/2)*height)*.95   #use for half of the cornea
    upper = (x_y_r[0]*height-x_y_r[2]*height)*.95
    lower = (x_y_r[0]*height+x_y_r[2]*height)*1.05
    box = (left, upper, right, lower)
    img = Image.open(ref_image)
    crop_ref_image = img.crop(box)
    if (show_image == True):
        plt.imshow(crop_ref_image)
    if (get_box == True):
        return box
    else:
        return crop_ref_image

# Return which of the unused images before the trial seems to represent a closed
# eyelid (darkest). This is used to start the trial time.
def get_last_blink(image_path):
    value = 100  # initialize to something big
    first_file = sorted(glob.glob(image_path))[0]
    ff_num = int(first_file.split(os.sep)[-1].split('_')[-1].split('.')[0])
    # check if on trial t10 vs t1-t9, extra character
    if (int(first_file.split(os.sep)[-2][1:]) == 10):
        end = 14
    else:
        end = 13
    for i in range(1, ff_num):
        first_unused = image_path[0:len(image_path)-5] + '_unused' + os.sep + first_file.split(
            os.sep)[-1][0:end] + get_padding(ff_num-i) + str(ff_num-i) + '.png'
        next_value = np.mean(np.array(Image.open(first_unused)))

        if (next_value < value):
            value = next_value
            imagenum = ff_num - i
    return imagenum

# Pad string rep of an integer with leading zeros (could be done with format strings)
def get_padding(num):
    if (num < 10):
        return '000'
    elif (num < 100):
        return '00'
    elif (num < 1000):
        return '0'
    else:
        return ''


###############################################################################
# Locating regions of interest (ROI)
###############################################################################

# Scan over an image with overlapping tiles, looking for those that exceed a TBU prediction 
# threshold. Returns an m-by-2 array of coordinates for the centers of the detected tiles.
def get_image_ROIs(ref_image, model_path, stride=32, thresh=.999, w=192, h=192, get_percent=False):
    model = load_model(model_path)
    ROI_centers = []
    centers = []
    pred_percent = []
    width, height = ref_image.size
    w, h = (192, 192)
    for col_i in range(0, width, stride):
        for row_i in range(0, height, stride):
            if (col_i + w < width and row_i + h < height):
                crop = ref_image.crop((col_i, row_i, col_i + w, row_i + h))
                re_sized = crop.resize((96, 96))
                re_sized = image.img_to_array(re_sized)
                # need to change it from (96,96,3) to (1,96,96,3):
                data = np.expand_dims(re_sized, axis=0)
                data = data/255
                pred_prob = model.predict(data, verbose=0)[0][-1]
                if pred_prob > thresh:
                    pred_percent.append((pred_prob))
                    centers.append((col_i, row_i))

    ROI_centers = np.zeros((len(centers), 2),dtype=int)
    for k in range(len(centers)):
        ROI_centers[k][0] = int(centers[k][0]+(w/2))
        ROI_centers[k][1] = int(centers[k][1]+(h/2))

    if get_percent == False:
        return ROI_centers
    else:
        return ROI_centers, pred_percent

# From the beginning of the trial, load 1 image for each second and scan for ROIs. Cluster the ROI 
# centers that are found, and if there are at least num_tbu clusters, stop and return the result.
# Outputs:
#    ROI_clust - m-by-3 array; each row is [x,y,cluster #] for an ROI
#    first_file - name of the first image file having at least num_tbu clusters
#    cropped_image - cropped image from first_file
def get_first_tbu(image_path, model_path, cor_doc, frame_rate, num_tbu=3, thresh=.999, v=3):
    files = sorted(glob.glob(image_path))
    clust_number = 0
    file_number = 55  #### CHANGE BACK TO ZERO
    while (clust_number < num_tbu and file_number < len(files)):
        cropped_image = get_crop(
            files[file_number], cor_doc, single_image=True, show_image=False, v=v)
        ROI_centers = get_image_ROIs(cropped_image, model_path, thresh=thresh)
        if (len(ROI_centers) > 1):
            clusters = hcluster.fclusterdata(ROI_centers, 75, criterion="distance")
            clust_number = max(clusters)
            ROI_clust = np.append(ROI_centers, clusters[..., None], 1)
        file_number = file_number + frame_rate

    first_file = files[file_number - frame_rate]
    return ROI_clust, first_file, cropped_image


# Turn each cluster into the most central w x h tile that represents it
# Returns a list of crop rectangles for the ROIs
def consolidate_ROI_clusters(ROI_centers, h=192, w=192):
    num_clusters = int(np.max(ROI_centers[:, 2]))
    boxes = []
    rows_delete = []
    for i in range(1, num_clusters+1):
        for j in range(len(ROI_centers)):
            if (ROI_centers[j, 2] != i):
                rows_delete.append((j))
        cluster_i = np.delete(ROI_centers, rows_delete, 0)
        center_x = (np.max(cluster_i[:, 0]) + np.min(cluster_i[:, 0]))/2
        center_y = (np.max(cluster_i[:, 1]) + np.min(cluster_i[:, 1]))/2
        right = center_x + (w/2)
        left = center_x - (w/2)
        upper = center_y - (h/2)
        lower = center_y + (h/2)
        boxes.append((left, upper, right, lower))
        rows_delete = []
    return boxes

###############################################################################
# Extracting intensity data
###############################################################################

# For a given image, find the location of the pixel in each box of interest that has minimum 
# intensity, after conversion to gray and applying a blur.
# Returns a list of pixel coordinates (one pixel per box).
def get_min_locations(file, cor_doc, boxes, sigma=5, v=3):
    locations = []
    kernel = int(np.ceil(4*sigma) + 1)
    ref_image = get_crop(file, cor_doc, single_image=True,
                         show_image=False, v=v)
    for i in range(len(boxes)):
        box_crop = ref_image.crop(boxes[i])
        # downsample from 192x192 to 96x96:
        re_sized = image.img_to_array(box_crop.resize((96, 96)))
        gray_image = cv2.cvtColor(re_sized, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(
            gray_image, (kernel, kernel), sigma, cv2.BORDER_REFLECT)
        r, c = np.where(blurred_image == blurred_image.min())
        locations.append([int(np.mean(r)), int(np.mean(c))])
    return locations

# Apply get_min_locations to each image in a trial.
# Returns list of (files x boxes) arrays of pixel locations, one per image.
def get_all_min_locations(image_path, cor_doc, boxes, v=3):
    files = sorted(glob.glob(image_path))
    loc = []
    for i in range(len(files)):
        locations = get_min_locations(files[i], cor_doc, boxes, v=v)
        loc.append(locations)

    locations = np.reshape(np.array(loc), (len(loc), 2))
    return locations

# Get the (blurred, gray) image intensity over the whole trial at each minimizing location in each 
# ROI box.
# The minlocs input is a (boxes x 2) array of locations relative to the ROI box.
# Returns an (images x boxes) array of intensity values.
def get_intensity(image_path, cor_doc, boxes,  minlocs, sigma=5, v=3):
    files = sorted(glob.glob(image_path))
    intensity = np.zeros((len(files), len(boxes)))
    kernel = int(np.ceil(4*sigma) + 1)
    for i in range(0, len(files)):
        ref_image = get_crop(
            files[i], cor_doc, single_image=True, show_image=False, v=v)
        for j in range(len(boxes)):
            gray_image = cv2.cvtColor(image.img_to_array(ref_image.crop(
                boxes[j]).resize((96, 96))), cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(
                gray_image, (kernel, kernel), sigma, cv2.BORDER_REFLECT)
            center_r = int(minlocs[j, 0])
            center_c = int(minlocs[j, 1])
            intensity[i, j] = blurred_image[center_r, center_c]
    return intensity

###############################################################################
# Predicting TBU probabilities
###############################################################################

# Use CNN to predict the probability of TBU in a single tile anchored at (x,y)
def get_proba_tbu(model_path, cor_doc, image_path, x, y, w=192, h=192, v=3):
    model = load_model(model_path)
    ref_image = get_crop(image_path, cor_doc,
                         single_image=True, show_image=False, v=v)
    crop = ref_image.crop((x, y, x + w, y + h))
    re_sized = image.img_to_array(crop.resize((96, 96)))
    # need to change it from (96,96,3) to (1,96,96,3)
    data = np.expand_dims(re_sized, axis=0)
    data = data/255
    return model.predict(data, verbose=0)[0][-1]

# Use CNN to predict the probability of TBU in each image for each ROI box
# Returns (images x boxes) array of probabilities
def get_proba_from_boxes(image_path, model_path, cor_doc, boxes, v=3):
    files = sorted(glob.glob(image_path))
    prob_tbu = np.zeros((len(files), len(boxes)))
    for i in range(0, len(files)):
        image = files[i]
        for j in range(len(boxes)):
            x, y = boxes[j][0], boxes[j][1]
            prob_tbu[i, j] = get_proba_tbu(
                model_path, cor_doc, image, x, y, v=v)
    return prob_tbu


###############################################################################
# Testing quality of the result
###############################################################################

# Compute fraction of the true min locations over time that lie outside of a small box around the 
# fixed nominal minimizer location
def test_location(locations, fixed_loc, tile_size=33):
    m = int((tile_size - 1)/2)
    fraction_inside = []
    for i in range(len(locations)):
        r,c = fixed_loc[i]
        pos = locations[i]
        above = np.where(pos[:, 0] < (r-m))[0]
        # delete the rows to avoid double counting when checking col
        pos = np.delete(pos, above, axis=0)
        below = np.where(pos[:, 0] > (r+m))[0]
        pos = np.delete(pos, below, axis=0)

        left = np.where(pos[:, 1] < (c-m))[0]
        pos = np.delete(pos, left, axis=0)
        right = np.where(pos[:, 1] > (c+m))[0]
        pos = np.delete(pos, right, axis=0)

        fraction_inside.append(len(pos)/len(locations[i]))
    return fraction_inside

 
def test_onset_intensity(intensity, prob_tbu, locations):
    onset = []
    for i in range(len(locations)):
        first = np.where( prob_tbu[:,i] > 0.999 )[0][0]
        onset.append(intensity[first][i])
    return onset

# Find relative decrease in the medians of the first 10% and last 10% of the trial intensity values.
def test_darkening(intensity):
    decrease = []
    m,n = intensity.shape
    for j in range(n):
        begin = np.median(intensity[:m//10, j])  # first 10 percent
        end = np.median(intensity[-m//10:, j])  # last 10 percent
        decrease.append((begin-end)/begin)
    return decrease

# adds time to the adv_pix array as col 0
# Inputs:
#    image_path - path to stablized directory
#    adv_pix - (len of images x num of boxes) array of adv pixels
#    frame_rate - frame rate of the stablized images
#    last_blink_image - last blink images to use as starting points (time = 0)
# Output:
#     data_new - updated adv_pix array with time col
def get_time_vector(image_path, frame_rate, last_blink_image):
    files = sorted(glob.glob(image_path))
    nfiles = len(files)
    first_file = files[0]
    first_num = int(first_file.split('/')[-1].split('_')[-1].split('.')[-2])
    new_col = np.zeros((nfiles, 1))
    for j in range(nfiles):
        new_col[j, 0] = ((1/frame_rate) * ((first_num-last_blink_image) + j))
    return new_col



