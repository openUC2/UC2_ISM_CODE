#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 23:01:17 2018

@author: bene
"""

import numpy as np
import cv2
import tifffile as tif
import matplotlib.pyplot as plt 
import tifffile as tif
from scipy.signal import find_peaks
import scipy.signal
from scipy.ndimage.filters import gaussian_filter
from skimage.feature import peak_local_max

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

    
import ismtools as ism
import utils as utils

import NanoImagingPack as nip

def readtif(my_videofile, key=1, scalesize=1):
    # this reads a frame from a tiff file and scales it - potentially avoid MP4 checkerboard
    myframe_raw = np.array(tif.imread(my_videofile,key=int(key)))
    myframe_raw = cv2.resize(myframe_raw, None, fx=scalesize, fy=scalesize, interpolation = cv2.INTER_CUBIC)
    return myframe_raw 

def readtifstack(my_videofile, scalesize=1, is_denoise=False):
    key=0
    mystack = []
    while(True):
        try:
            # this reads a frame from a tiff file and scales it - potentially avoid MP4 checkerboard
            myframe_raw = np.array(tif.imread(my_videofile,key=int(key)))
            if is_denoise:
                myframe_raw = denoise_wavelet(myframe_raw, multichannel=False, rescale_sigma=True)
            mystack.append(cv2.resize(myframe_raw, None, fx=scalesize, fy=scalesize, interpolation = cv2.INTER_CUBIC))
            key+=1
        except(IndexError):
            print('End of file is reached @ ' + str(key))
            break    
        
    return np.array(mystack)

def reducetime(ismstack, is_debug=False):
    
    diff_time = np.sum(np.abs(ismstack[0:-2,:,:]-ismstack[1:-1,:,:]), axis=(1,2))
#    diff_time = np.squeeze(cv2.GaussianBlur(diff_time, (7, 1), 0))
    diff_time = gaussian_filter(diff_time*1., sigma=7)    
        
    # detect peaks - each peak corresponds to one change of the illuminatino pattern in Y 
    peaks, _ = find_peaks(np.squeeze(diff_time), distance=5)
    
    if(is_debug): 
        plt.plot(diff_time)
        plt.plot(peaks, diff_time[peaks], "x")
        plt.show()
    
    ismstack_red = np.zeros((len(peaks)+1,ismstack.shape[1], ismstack.shape[2]))
    timestep = 0
    frames_per_timestep = 0
    for i in range(ismstack.shape[0]):
        ismstack_red[timestep,:,:]+=ismstack[i,:,:,]
        frames_per_timestep = frames_per_timestep +1
        if(np.isin(i, peaks)):
            ismstack_red[timestep,:,:] = np.float32(ismstack_red[timestep,:,:])/frames_per_timestep
            frames_per_timestep = 0
            timestep = timestep+1
    ismstack_red[timestep,:,:] = ismstack_red[timestep,:,:]/frames_per_timestep        
            
    ismstack_red = ismstack_red/np.expand_dims(np.expand_dims(np.mean(ismstack_red,(1,2)),-1),-1)
    
    print('Final number of Timesteps: '+str(ismstack.shape[0]) +' vs. ' +str(ismstack_red.shape[0]))
        
    return ismstack_red

def get_peaks_from_image(myframe, mindist=15, blurkernel = 2,  is_debug=True):
    # This gets the approximate peaks from the image
    mythresh = np.mean(myframe)
    myframe_blurred = nip.gaussf(myframe, blurkernel)
    myxycoords = peak_local_max(myframe_blurred, min_distance=mindist,threshold_abs=mythresh)

    mypeaks = np.zeros(myframe.shape)
    mypeaks[myxycoords[:,0],myxycoords[:,1]]=1
    
    # visualize the peaks
    if is_debug:
        plt.figure()
        plt.subplot(131), plt.plot(myxycoords[:,1],myxycoords[:,0],'x'), plt.title('Detected Peaks'), 
        plt.subplot(132), plt.imshow(nip.gaussf(np.float32(mypeaks/np.float32(np.max(mypeaks + mypeaks))),1), cmap='hot')
        plt.subplot(133), plt.imshow(np.float32(myframe), cmap='hot')
        plt.show()


    if 0:
        # check for ordering
        # using some dummy data for this example
        xs = mypeaks_pos[:,0]
        ys = mypeaks_pos[:,1]
        
        # 'bo-' means blue color, round points, solid lines
        plt.plot(xs,ys,'x')
        
        # zip joins x and y coordinates in pairs
        iiter = 0
        for x,y in zip(xs,ys):
            
        
            label = str(iiter)
        
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
            iiter += 1

    return mypeaks, myxycoords


def adjust_grating_constant(my_fit_grid, g_diff_thresh = 2, is_debug=False):
    #    g_diff_thresh = 2 how many pixels deviation of the original grating constant do we allow?
    # correct for mismatching grating period along vertical direction
    sample_grid = np.arange(my_fit_grid.shape[0])       # provide some sampling for fitting function
    g_fit_grid_raw = np.mean(my_fit_grid,1)             # now get the line from the actually fitted 2D grid..
    
    g_fit_grid_par = np.polyfit(sample_grid,g_fit_grid_raw , 1) # fit a line to the grid-spacing with grating constant g and offst
    g_fit_grid = np.poly1d(g_fit_grid_par)              # sample the fitted data
    #g_fft_grid = my_g_grid*sample_grid                  # get the same line from the fourier specrum
    
    g_fit_grid_sampled = g_fit_grid(sample_grid)        # sample the fitted data (from previous sampling..)
    g_fit_diff = (g_fit_grid_raw-g_fit_grid_sampled)    # find difference
    g_fit_diff -= np.mean(g_fit_diff)                   # subtract background/offset
    g_needchange = np.abs(g_fit_diff)>g_diff_thresh     # find candidates which need to be adjusted

    # adjusting the spacing 
    my_fit_grid[g_needchange,:] -= np.expand_dims(g_fit_grid_raw[g_needchange]- g_fit_grid_sampled[g_needchange],1)
    g_fit_grid_raw_new = np.mean(my_fit_grid,1)             # now get the line from the actually fitted 2D grid..
    
    if is_debug:
        plt.title('spacing between lines (before)'), plt.plot(g_fit_grid_raw)
        plt.title('spacing between lines (after)'), plt.plot(g_fit_grid_raw_new)
        plt.title('sampling of lines'), plt.plot(g_fit_grid_sampled)
        #plt.title('spacing between lines (FFT spectrum)'), plt.plot(g_fft_grid)
        plt.title('spacing between lines (fit)'), plt.plot(g_fit_diff)
        plt.title('These lines need a change'), plt.plot(g_needchange*np.max(g_fit_grid_raw),'x')
        plt.show()
        
    return my_fit_grid

def fit_illpattern_ism(mypeaks_pos, polyn = 2, max_std = 2., searchdist_vert = 6, searchdist_horz = 6, is_debug =False, is_fillmissing=True, g_vert=None, g_horz=None):
    ''' 
    This is a funciton which extracts the lines from an ISM frame 
    mypeaks_pos - peaks from the peak find routine (XY positions)
    polyn  - order of the polynome which is used to fit the line 
    max_std - standard deviation to remove outliers 
    searchdist - minimum distance between possible lines 
    '''
    #%%
    min_pos_vert = np.min(mypeaks_pos[:,1])-searchdist_vert
    max_pos_vert = np.max(mypeaks_pos[:,1])+searchdist_vert
    min_pos_horz = np.min(mypeaks_pos[:,0])-searchdist_horz
    max_pos_horz = np.max(mypeaks_pos[:,0])+searchdist_horz
    
    vert_eval = np.linspace(min_pos_vert,max_pos_vert,4*(max_pos_vert-min_pos_vert))
    horz_eval = np.linspace(min_pos_horz,max_pos_horz,4*(max_pos_horz-min_pos_horz))
    
    my_fit_vert = [] # nonlinear fit of deformed pattern
    my_fit_horz = []
    
    # iterate over vertical lines 
    for i_vert in range(min_pos_vert ,max_pos_vert , searchdist_vert):
        try: 
            # select the peaks inside a range (upper/lower boundary )
            my_peaks_in_range =  mypeaks_pos[((mypeaks_pos[:,0]>=(i_vert-searchdist_vert )) * (mypeaks_pos[:,0]<=(i_vert+searchdist_vert ))),:]
            
            # skip if no peaks are found! 
            if my_peaks_in_range.shape[0]>0:
                # sort the peaks
                my_peaks_in_range = my_peaks_in_range[np.argsort(my_peaks_in_range[:,1]),:]
                
                # add point at the edges of the line 
                boundary_val_y_min = np.array([[np.mean(my_peaks_in_range,axis=0)[0],min_pos_vert]])
                boundary_val_y_max = np.array([[np.mean(my_peaks_in_range,axis=0)[0],max_pos_vert]])        
                my_peaks_in_range = np.vstack((boundary_val_y_min, my_peaks_in_range, boundary_val_y_max))
        
                # check for outliers outliers
                my_std = np.std(my_peaks_in_range[:,0])
        
                # only consider a line with more than 3 potential members
                if my_peaks_in_range.shape[0]>3 and my_std<max_std:
                    
                    # fit polynomial function to the points 
                    my_peaks_in_range_fit = np.poly1d(np.polyfit(my_peaks_in_range[:,1], my_peaks_in_range[:,0], polyn)) 
                    
                    # evaluate the result - sample it!
                    my_peaks_in_range_fit_eval = my_peaks_in_range_fit(vert_eval)
                    
                    # check again if the stdv is still fine 
                    my_std = np.std(my_peaks_in_range_fit_eval)
                    
                    if(my_std<max_std):
                        # only add the line to the selection if the quality is ok
                        
                        if(len(my_fit_vert)>0): 
                            # make sure, that the same line is not added twice
                            if(is_debug): print(np.abs(np.mean(my_fit_vert[-1]-my_peaks_in_range_fit_eval)))
                            if np.abs(np.mean(my_fit_vert[-1]-my_peaks_in_range_fit_eval)) < (searchdist_vert):
                                # if it's roughly the same, skip this line
                                if(is_debug):  print(' Looks like a double-fit was detected')
                            else:
                                my_fit_vert.append(my_peaks_in_range_fit_eval)    
                        else:
                            my_fit_vert.append(my_peaks_in_range_fit_eval)
                           
                    else:
                        if is_debug: print("skipping line due to high stdv..")
        
                    if(is_debug):
                        plt.plot(my_peaks_in_range[:,0], my_peaks_in_range[:,1], 'x'), plt.axis('equal')
                        plt.plot(my_peaks_in_range_fit_eval, vert_eval, '-'), plt.axis('equal')
                        plt.xlim(right=max_pos_vert) #xmax is your value
                        plt.xlim(left=min_pos_vert) #xmin is your value
                        plt.ylim(top=max_pos_horz) #ymax is your value
                        plt.ylim(bottom=min_pos_horz) #ymin is your value
        except(ValueError):
            if(is_debug): print('Err')
        except(TypeError):
           if(is_debug):  print('Err')
            
           
            
    #%% now  we want to add missing grating components 
    # difference between neighbouring peaks -> we want to have the lower "mean" to be the grating constant of our graitng
    my_fit_vert = np.array(my_fit_vert)
    p_grating_vert_estimate = np.mean(my_fit_vert,-1) # position of grating
    d_grating_vert_estimate = p_grating_vert_estimate[1:]-p_grating_vert_estimate[:-1] # distnace between gratings
    g_grating_vert_estimate = np.mean(utils.reject_outliers(d_grating_vert_estimate, m=1))
    if np.isnan(g_grating_vert_estimate): g_grating_vert_estimate = np.mean(np.array(g_vert))
    if is_debug: print("Grating Constant (vertical) is: "+str(g_grating_vert_estimate) )
    if is_debug: 
        for i in range(0, my_fit_vert.shape[0]): plt.plot(my_fit_vert[i,:])
        plt.show()
    
    
    #%% filling missing grating positions with lines
    iiter_vert = 0
    iiter_vert_missing = 0 # internal counter
    my_fit_vert_new = []
    #for i in range(0, my_fit_vert.shape[0]): plt.plot(my_fit_vert[i,:]); plt.show()
    while True:
        # see if there is a grating near the current one
        if (iiter_vert+1) >= p_grating_vert_estimate.shape[0]: break 

        if((p_grating_vert_estimate[iiter_vert+1] - p_grating_vert_estimate[iiter_vert] - iiter_vert_missing*g_grating_vert_estimate)>(g_grating_vert_estimate*1.8)): # see if there is a missing line in the grid
            iiter_vert_missing += 1
            if is_debug: print("Changed: "+ str(iiter_vert_missing)+"/"+str(iiter_vert))
            # add a line with a distance of the grating constant after the current one - we will copy the coordinates from the previous line (it's simpler..)
            my_fit_vert_iter_new = my_fit_vert[iiter_vert]  + (iiter_vert_missing)*g_grating_vert_estimate
            my_fit_vert_new.append(my_fit_vert_iter_new)
        else:
            iiter_vert_missing = 0 # reset internal counter 
            my_fit_vert_new.append(my_fit_vert[iiter_vert,:]) # proceed with ordinary line
            iiter_vert += 1
            

    if is_fillmissing: 
        my_fit_vert =  np.array(my_fit_vert_new)
        
    if False:# is_debug: 
        for i in range(0, my_fit_vert_new.shape[0]): plt.plot(my_fit_vert_new[i,:]),    plt.show()
    

    #%%
    
        
    # iterate over horizontal lines 
    for i_vert in range(min_pos_horz ,max_pos_horz ,searchdist_horz ):
        try: 
            # select the peaks inside a range (upper/lower boundary )
            my_peaks_in_range =  mypeaks_pos[((mypeaks_pos[:,1]>=(i_vert-searchdist_horz )) * (mypeaks_pos[:,1]<=(i_vert+searchdist_horz ))),:]
                
            if my_peaks_in_range.shape[0]>0:           

                # sort the peaks
                my_peaks_in_range = my_peaks_in_range[np.argsort(my_peaks_in_range[:,0]),:]
                
                # add point at the edges of the line 
                boundary_val_y_min = np.array([[min_pos_horz, np.mean(my_peaks_in_range,axis=0)[1]]])
                boundary_val_y_max = np.array([[max_pos_horz, np.mean(my_peaks_in_range,axis=0)[1]]])        
                my_peaks_in_range = np.vstack((boundary_val_y_min, my_peaks_in_range, boundary_val_y_max))
        
                # check for outliers outliers
                my_std = np.std(my_peaks_in_range[:,1])
        
                if my_peaks_in_range.shape[0]>3 and my_std<max_std:
                    
                    # fit polynomial function to the points 
                    my_peaks_in_range_fit = np.poly1d(np.polyfit(my_peaks_in_range[:,0], my_peaks_in_range[:,1], polyn)) 
                    my_peaks_in_range_linfit = np.poly1d(np.polyfit(my_peaks_in_range[:,0], my_peaks_in_range[:,1], 0)) 
                    # evaluate the result - sample it!
                    my_peaks_in_range_fit_eval = my_peaks_in_range_fit(horz_eval)
                    my_peaks_in_range_linfit_eval = my_peaks_in_range_linfit(horz_eval)
    
                    
                    # check again if the stdv is still fine 
                    my_std = np.std(my_peaks_in_range_fit_eval)
                    if(my_std<max_std):
                        # only add the line to the selection if the quality is ok
                        if(len(my_fit_horz)>0): 
                            # make sure, that the same line is not added twice
                            if np.abs(np.mean(my_fit_horz[-1]-my_peaks_in_range_fit_eval)) < (searchdist_horz):
                                # if it's roughly the same, skip this line
                                if(is_debug): print(' Looks like a double-fit was detected')
                            else:
                                my_fit_horz.append(my_peaks_in_range_fit_eval)        
                        else:
                            my_fit_horz.append(my_peaks_in_range_fit_eval)
    
                    if(is_debug):
                        plt.plot(my_peaks_in_range[:,0], my_peaks_in_range[:,1], 'x'), plt.axis('equal')
                        plt.plot(horz_eval, my_peaks_in_range_fit_eval, '-'), plt.axis('equal')
                        plt.xlim(right=max_pos_vert) #xmax is your value
                        plt.xlim(left=min_pos_vert) #xmin is your value
                        plt.ylim(top=max_pos_horz) #ymax is your value
                        plt.ylim(bottom=min_pos_horz) #ymin is your value                        
        except(ValueError):
            if(is_debug): print('Err')
        except(TypeError):
            if(is_debug): print('Err')
    
    
    
    #%% now  we want to add missing grating components 
    # difference between neighbouring peaks -> we want to have the lower "mean" to be the grating constant of our graitng
    my_fit_horz = np.array(my_fit_horz)
    p_grating_horz_estimate = np.mean(my_fit_horz,-1) # position of grating
    d_grating_horz_estimate = p_grating_horz_estimate[1:]-p_grating_horz_estimate[:-1] # distnace between gratings
    g_grating_horz_estimate = np.mean(utils.reject_outliers(d_grating_horz_estimate, m=1))
    
    if np.isnan(g_grating_horz_estimate): g_grating_horz_estimate = np.mean(np.array(g_horz))
    if is_debug: print("Grating Constant (horzical) is: "+str(g_grating_horz_estimate) )
    if is_debug: 
        for i in range(0, my_fit_horz.shape[0]): plt.plot(my_fit_horz[i,:])
        plt.show()
    
    
    
     #%% filling missing grating positions with lines
    iiter_horz = 0
    iiter_horz_missing = 0 # internal counter
    my_fit_horz_new = []
    #for i in range(0, my_fit_horz.shape[0]): plt.plot(my_fit_horz[i,:]); plt.show()
    while True:
        # see if there is a grating near the current one
        if (iiter_horz+1) >= p_grating_horz_estimate.shape[0]: break 

        if((p_grating_horz_estimate[iiter_horz+1] - p_grating_horz_estimate[iiter_horz] - iiter_horz_missing*g_grating_horz_estimate)>(g_grating_horz_estimate*1.8)): # see if there is a missing line in the grid
            iiter_horz_missing += 1
            if is_debug: print("Changed: "+ str(iiter_horz_missing)+"/"+str(iiter_horz))
            # add a line with a distance of the grating constant after the current one - we will copy the coordinates from the previous line (it's simpler..)
            my_fit_horz_iter_new = my_fit_horz[iiter_horz]  + (iiter_horz_missing)*g_grating_horz_estimate
            my_fit_horz_new.append(my_fit_horz_iter_new)
        else:
            iiter_horz_missing = 0 # reset internal counter 
            my_fit_horz_new.append(my_fit_horz[iiter_horz,:]) # proceed with ordinary line
            iiter_horz += 1
            

    
    if is_fillmissing: 
        my_fit_horz =  np.array(my_fit_horz_new)
    if False:# is_debug: 
        for i in range(0, my_fit_horz_new.shape[0]): plt.plot(my_fit_horz_new[i,:]),    plt.show()
    
  
    
    if(is_debug):
        plt.xlim(min_pos_vert, max_pos_vert)
        plt.ylim(min_pos_horz, max_pos_horz)
        plt.show()

    return my_fit_vert, vert_eval, my_fit_horz, horz_eval, g_grating_vert_estimate, g_grating_horz_estimate


def generate_illumination_grid(test_frame, my_fit_vert, my_vert, my_fit_horz, my_horz, is_debug=False):
    #%
    # now we need to find the intersecting points of the grid 
    # since we are very lazy, we do that on an upsampled  integer grid 
    my_grid = np.zeros((test_frame.shape[0], test_frame.shape[1]))
    
    # resample to original grid - hacky, I know
    #my_fit_vert = nip.resample(my_fit_vert,(1,test_frame.shape[1]/my_fit_vert.shape[1]))
    #my_vert  = nip.resample(my_vert,(1,test_frame.shape[1]/my_vert.shape[0]))
    
    # draw for vertical lines
    for i in range(0, my_fit_vert.shape[0]): 
        my_index_m,my_index_n = np.uint32(np.abs(my_fit_vert[i,:])),np.uint32(np.abs(my_vert))
        my_index_tmp = my_index_m.copy()
        my_index_m = my_index_m[my_index_tmp<my_grid.shape[1]] # hacky, but should cleanup index problems
        my_index_n = my_index_n[my_index_tmp<my_grid.shape[1]]
        try:
            my_grid[my_index_m,my_index_n]+=1
        except:
            if is_debug: print('Index out of bounds..')
                
     # resample to original grid
    #my_fit_horz = nip.resample(my_fit_horz,(1,test_frame.shape[1]/my_fit_horz.shape[1]))
    #my_horz  = nip.resample(my_horz,(1,test_frame.shape[1]/my_horz.shape[0]))
       
    # draw for horizontal lines
    for i in range(0, my_fit_horz.shape[0]): 
        my_index_n,my_index_m = np.int32((np.abs(my_fit_horz[i,:]))),np.int32((np.abs(my_horz)))
        my_index_tmp = my_index_m.copy()
        my_index_m = my_index_m[my_index_tmp<my_grid.shape[1]] # hacky, but should cleanup index problems
        my_index_n = my_index_n[my_index_tmp<my_grid.shape[1]]
        try:
            my_grid[my_index_m,my_index_n]+=1
        except:
            if is_debug: print('Index out of bounds..')

    
    my_grid_index = np.where(my_grid>1) 
    my_grid = my_grid > 1
    return my_grid, my_grid_index 


''' -------- SHIFT TOOLS --------'''


def draw_lines(img, houghLines, color=[0, 255, 0], thickness=2):
    for line in houghLines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
 
            cv2.line(img,(x1,y1),(x2,y2),color,thickness)   
                
 
def weighted_img(img, initial_img, alpha=0.8, beta=1., lambda0=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lambda0) 

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    image = np.uint8(image/np.max(image)*(2**8-1))
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edged = cv2.Canny(np.uint8(image), lower, upper)
 
    # return the edged image
    return edged

def FindHoughLinesP(inputimage, lowerbound = 20, blur=13):
    #%% Here we try to estimate the shift along x 
    gray_image = np.uint8(inputimage/np.max(inputimage)*(2**8-1))
    blurred_image = cv2.GaussianBlur(gray_image, (blur, blur), 0)
    edges_image = cv2.Canny(blurred_image, lowerbound, 120)
    plt.imshow(edges_image), plt.show()
    # Get the Hough transform and select the dominant lines   
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 60  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    
    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges_image, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    
    index=0
    m=np.zeros(lines.shape[0])
    for line in lines:
        for x1,y1,x2,y2 in line:
            m[index] = (x1-x2)/(y1-y2)
            index=index+1
            cv2.line(gray_image,(x1,y1),(x2,y2),(255,0,0),1)
    mym = np.mean(m)
    
    return mym


def get_shift_coordinates(my_videofile, scalesize, is_debug = False):
    '''
    my_videofile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/CheapConfocal/MATLAB/P12-HWP9_2018-11-1514.28.24.mp4.tif'
    scalesize = 1;
    is_debug = True
    get_shift_coordinates(my_videofile, scalesize)
    '''
    # 1.) Read all frames 
    framelist = []
    for i in range(10000):
        try:
            # read the first frame
            framelist.append(readtif(my_videofile, key=i, scalesize=scalesize))
            if(is_debug): print(str(i))
        except(IndexError):
            print('End of file is reached @ ' + str(i))
            break
            
        
    #%% 2.) project the time-stack along X and find the shift of the pupils using Hough Transforms
    # convert to array
    framelist = np.array(framelist)    
    
    # get X-projection    
    proj_x = np.std(framelist, axis=2)
    if(is_debug): plt.imshow(proj_x)

    # find Hough Transform and get their slope
    mymx = FindHoughLinesP(proj_x)
    
    #%% 3.) project the time-stack along Y and find the shift of the pupils using Hough Transforms
    # get Y-projection    
    proj_y = np.std(framelist, axis=1)
    proj_y = np.log(proj_y**7)
    myhighpassfilter = np.fft.fftshift(np.squeeze(rr(inputsize_x=proj_y.shape[0], inputsize_y=proj_y.shape[1], inputsize_z=1)>(np.min(proj_y.shape)*.1)))
    proj_y = np.real(np.fft.ifft2(np.fft.fft2(proj_y)*myhighpassfilter))
    if(is_debug): plt.imshow(myhighpassfilter),  plt.show()
    if(is_debug): plt.imshow(proj_y), plt.show()
    if(is_debug): plt.imshow(proj_y), plt.show()
    
    # estimate rotation by projecting along y and rotating => max == rotation?
    mymy = -np.inf
    mygrady = 0
    for iangle in np.linspace(-5,5,40):
        img = proj_y
        rows,cols = img.shape
    
        M = cv2.getRotationMatrix2D((cols/2,rows/2),iangle,1)
        img = cv2.warpAffine(img,M,(cols,rows))

        imgsum = np.sum(img, 0)
        mygrad = np.abs(imgsum[0:-2]-imgsum[1:-1])
        if np.mean(mygrad) > mygrady:
            mygrady = np.mean(mygrad)
            mymy = -np.tan(((iangle+90)/90)*np.pi)
        if(is_debug): plt.title('rotated projection along y'), plt.imshow(img), plt.show()
        #if(is_debug): plt.title('finite difference of prjection along y'), plt.plot(mygrad), plt.show()
        print('Gradl @ ' +str(iangle) + ' is '+str(np.mean(mygrad)) )

    # MAGIC
    proj_y_grad = np.std(proj_y, axis=1)
    plt.plot(proj_y_grad), plt.show()
    
    finite_diff_y = np.abs(proj_y[0:-2,:] - proj_y[1:-1,:])# -np.roll(proj_y, [1], axis=0))
    if(is_debug): plt.imshow(finite_diff_y), plt.show()
    # Still ugly to find the steps in Y-Shifts 
    mysteps = np.sum(finite_diff_y, axis=1)
    mysteps = np.squeeze(cv2.GaussianBlur(mysteps, (1, 15), 0))

    # detect peaks - each peak corresponds to one change of the illuminatino pattern in Y 
    peaks, _ = find_peaks(np.squeeze(mysteps), distance=20)
    if(is_debug): 
        plt.plot(mysteps)
        plt.plot(peaks, mysteps[peaks], "x")
        plt.show()
    my_y_shifts = peaks-1
  
    #%% 4.) Write out the shift-coordinates for the illumination pattern
    myshift=np.zeros((proj_x.shape[0],2))
    yshift = 0
    xshift_offset = 1
    for i in range(proj_x.shape[0]):
        # linear shift over all frames
        xshift= np.floor(i*mymx) - xshift_offset
        
        if(np.isin(i, my_y_shifts)):
            yshift = yshift+1
            xshift_offset = xshift_offset + xshift;
            
        yshift = np.floor(mymy/2*i)
        myshift[i,:] = (xshift, yshift)
        
    plt.title('Found Shift positions')
    plt.plot(myshift[:,0], myshift[:,1], 'x')
    
    return myshift

def fit_line(ydata, is_debug=False):
    from scipy.optimize import curve_fit
    def func(x, m, b):
        return m*x+b
    xdata = range(ydata.shape[0])
    popt_x, pcov_x = curve_fit(func, xdata, ydata)
        
    if(is_debug):
        plt.plot(func(xdata, *popt_x))
        plt.plot(ydata, 'o')
        plt.show()
    return func(xdata, *popt_x)


def rr(inputsize_x=100, inputsize_y=100, inputsize_z=100, x_center=0, y_center = 0, z_center=0):
    x = np.linspace(-inputsize_x/2,inputsize_x/2, inputsize_x)
    y = np.linspace(-inputsize_y/2,inputsize_y/2, inputsize_y)

    if inputsize_z<=1:
        xx, yy = np.meshgrid(x+x_center, y+y_center)
        r = np.sqrt(xx**2+yy**2)
        r = np.transpose(r, [1, 0]) #???? why that?!
    else:
        z = np.linspace(-inputsize_z/2,inputsize_z/2, inputsize_z)
        xx, yy, zz = np.meshgrid(x+x_center, y+y_center, z+z_center)
        xx, yy, zz = np.meshgrid(x, y, z)
        r = np.sqrt(xx**2+yy**2+zz**2)
        r = np.transpose(r, [1, 0, 2]) #???? why that?!
        
    return np.squeeze(r)



def find_shift_grid(ismstack,mygrid_raw, cropsize):
    print('find accumulated shift between frames in ISM stack')
    myshifts_list = []
    mygrid_last = mygrid_raw
    for i in range(0,ismstack.shape[0]):
        # find the shift 
        current_frame = ismstack[i,:,:]
        current_frame = utils.extract2D(current_frame, cropsize)
        global_shift = utils.find_shift_lattice(mygrid_last,current_frame)
        mygrid = np.zeros(mygrid_last.shape)
        mygrid = np.roll(1*mygrid_last, -int(global_shift[0]), axis=0)
        mygrid = np.roll(mygrid, -int(global_shift[1]), axis=1)
        mygrid_last = mygrid
        if(i==0):
            myshifts_list.append(global_shift)
        else:
            myshifts_list.append(global_shift+myshifts_list[i-1])
    myshifts = np.array(myshifts_list)    
    if(True):
        # sometimes there is a wrap already detected by this procedure - first: unwrap it!
        unwrapx_peaks = np.abs(myshifts[0:-2,0]-myshifts[1:-1,0])           
        unwrapx_peaks = np.roll((unwrapx_peaks > (np.max(unwrapx_peaks)*.5)),1,0)
        unwrapx_peaks = np.where(unwrapx_peaks>0)
        
        for i in range(myshifts.shape[0]):
            if(np.isin(i, unwrapx_peaks)):
                myshifts[i:-1,0] = myshifts[i:-1,0] + np.abs(myshifts[i-1,0]-myshifts[i,0])
                print(np.abs(myshifts[i-1,0]-myshifts[i,0]))
                
    # fit straight line to the shift            
    myshifts[:,0] = ism.fit_line(myshifts[:,0], True)

    return(myshifts)

def compute_superconfocal(ismstack, is_debug):
    print('First we want to produce a super-confocal image R. Heintzman et al 2006')
    superconfocal = np.max(ismstack, axis=0)+np.min(ismstack,axis=0)-2*np.mean(ismstack,axis=(0))
    
    if is_debug:
        plt.subplot(121), plt.title('Superconfocal'), plt.imshow(superconfocal, cmap='gray'), plt.colorbar(), plt.show()
    return superconfocal

def compute_brightfield(ismstack, is_debug):
    print('We compute the BF equivalent as the projection of the stack')
    bf = np.mean(ismstack, axis=0) 
    
    if is_debug:
        plt.subplot(121), plt.title('Brightfield'), plt.imshow(bf, cmap='gray'), plt.colorbar(), plt.show()
    return bf


def estimate_ism_pars(testframe, mindist, radius_ft=None, is_debug=False):
    '''
    Estimates the rotation and grating constant of the ISM grid
    
    Parameters
    ----------
    testframe : 2D numpy array (raw ISM frame)
        DESCRIPTION.
    mindist : minimum distance for peak detection (between 2 illuminating spots)
        DESCRIPTION.
    radius_ft : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    searchdist : TYPE
        DESCRIPTION.
    rottheta : TYPE
        DESCRIPTION.

    '''
    print('Estimating the peaks from a single frame.')
    mysize = testframe.shape[1]    # size from the original 
    
    if radius_ft is None:
        radius_ft = .03
        print("Setting the filter size to: "+str(radius_ft))
        
        
    # estimate grating constant 
    myismspectrum = np.log(1+np.abs(nip.ft(testframe)*(nip.rr(testframe,freq='ftfreq')>radius_ft)))
    myismspectrum_thresh=myismspectrum.copy()
    myismspectrum_thresh[myismspectrum < (np.max(myismspectrum)*.9)]=0
    mypeaks , mypeaks_pos = get_peaks_from_image(myismspectrum_thresh, mindist=mindist, blurkernel = 4, is_debug=is_debug)
    
    # choose peaks with highest power
    mymaxpeakpos = np.where((myismspectrum_thresh * mypeaks)>np.mean(myismspectrum_thresh[(myismspectrum_thresh * mypeaks)>1]))
    
    # find grating constants and rotation angle
    
    # rearrange coordinates
    index_y_1 = mymaxpeakpos[0][0] 
    index_y_2 = mymaxpeakpos[0][1]
    index_x_1 = mymaxpeakpos[1][0]
    index_x_2 = mymaxpeakpos[1][1]

    # estimate grating
    d_x = index_x_1 - index_x_2     # distance between x coordinates 
    d_y = index_y_1 - index_y_2     # distance between x coordinates 
    dr_x = np.sqrt(d_x**2+d_x**2)   # absolute distnace in x
    dr_y = np.sqrt(d_y**2+d_y**2)   # absolute distnace in y (e.g. pythagoras)
    
    # estimate rotation
    rottheta = -(90+np.arctan2(d_x,d_y)/np.pi*180)
    
    # grating periods
    g_x = (mysize)/(dr_x/2) # grating constant for x as nyquist sampling
    g_y = (mysize)/(dr_y/2) # grating constant for y as nyquist sampling
    
    myg = np.min((g_x,g_y))
    print('Gratingconstant: '+str(myg))
    print('rotation : '+str(rottheta))
    
    # assign values for more robust grating search
    # adjust searchdistance
    searchdist = int(myg*.5) # minimum seperation between two different lines (vertically)
    
    plt.subplot(121), plt.title('spectrum of pat'),plt.imshow(myismspectrum, cmap='gray')
    plt.subplot(122), plt.title('detected peaks'), plt.plot(mymaxpeakpos,'x')
    return searchdist, rottheta
