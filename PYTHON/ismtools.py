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

import numpy as np
import ismtools as ism
import utils as utils



def readtif(my_videofile, key=1, scalesize=1):
    # this reads a frame from a tiff file and scales it - potentially avoid MP4 checkerboard
    myframe_raw = np.array(tif.imread(my_videofile,key=int(key)))
    myframe_raw = cv2.resize(myframe_raw, None, fx=scalesize, fy=scalesize, interpolation = cv2.INTER_CUBIC)
    return myframe_raw 

def readtifstack(my_videofile, scalesize=1):
    key=0
    mystack = []
    while(True):
        try:
            # this reads a frame from a tiff file and scales it - potentially avoid MP4 checkerboard
            myframe_raw = np.array(tif.imread(my_videofile,key=int(key)))
            mystack.append(cv2.resize(myframe_raw, None, fx=scalesize, fy=scalesize, interpolation = cv2.INTER_CUBIC))
            key+=1
        except(IndexError):
            print('End of file is reached @ ' + str(key))
            break    
    return np.array(mystack)

def reducetime(ismstack, debug=False):
    
    diff_time = np.sum(np.abs(ismstack[0:-2,:,:]-ismstack[1:-1,:,:]), axis=(1,2))
    diff_time = np.squeeze(cv2.GaussianBlur(diff_time, (7, 1), 0))
        
    # detect peaks - each peak corresponds to one change of the illuminatino pattern in Y 
    peaks, _ = find_peaks(np.squeeze(diff_time), distance=5)
    
    if(debug): 
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
    return ismstack_red


def get_peaks_from_image(myframe, mindist=15):
    # This gets the approximate peaks from the image
    #myframe = cv2.blur(1.*myframe, (3, 3))
    mythresh = np.mean(myframe)
    from skimage.feature import peak_local_max
    xy = peak_local_max(myframe, min_distance=mindist,threshold_abs=mythresh)
    plt.plot(xy[:,1],xy[:,0],'x'), plt.title('Detected Peaks'), plt.show()
    mypeaks = np.zeros(myframe.shape)
    mypeaks[xy[:,0],xy[:,1]]=1
    #myframe_test = gaussian_filter(myframe*mymask, sigma=1) # cv2.blur(myframe*mymask, (3,3))
    #mypeaks_raw = utils.detect_peaks(myframe_test)
    return mypeaks


def get_perfect_lattice(mypeaks_raw, cropsize):
    # work on new size
    mypeaks = utils.extract2D(1*mypeaks_raw, cropsize)*100
    stripesize = 3
    Niter = 8
    stripalong_y = np.zeros((cropsize[0], stripesize*2, Niter))
    stripalong_x = np.zeros((stripesize*2, cropsize[1], Niter))
    
    # perform sumprojection allong X/Y
    sum_x = np.sum(mypeaks, axis=0)
    sum_y = np.sum(mypeaks, axis=1)
    
    # and filter out the borders - we can not crop out the borders anyway!
    sum_x[0:stripesize,]=0
    sum_x[-stripesize-1:-1,]=0
    sum_y[0:stripesize,]=0
    sum_y[-stripesize-1:-1,]=0
       
    mypeaks_cutout = mypeaks.copy()
    for i in range(Niter):
        try:
                
            max_x = np.argmax(sum_x)
            max_y = np.argmax(sum_y)
            
            # cut out a stripe along X/Y around the max-lines
            stripalong_y[:,:,i] = mypeaks[:,max_x-stripesize:max_x+stripesize]
            stripalong_x[:,:,i] = mypeaks[max_y-stripesize:max_y+stripesize,:]
        
            # make sure you'Re not using it again! 
            mypeaks_cutout[:,max_x-stripesize:max_x+stripesize] = -10
            mypeaks_cutout[max_y-stripesize:max_y+stripesize,:] = -10
            #plt.imshow(mypeaks_cutout), plt.show()
        
            sum_x[max_x-stripesize:max_x+stripesize] = -10
            sum_y[max_y-stripesize:max_y+stripesize] = -10
        
            #plt.imshow(stripalong_x[:,:,i]), plt.show()
            #plt.imshow(stripalong_y[:,:,i]), plt.show()
        except(ValueError):
            print('Error in fitting the stripes')
    
    #%%
    print('# Observe Peaks along X')
    myft_peak_x = np.zeros((cropsize[0], ))
    for i in range(Niter):
        #plt.imshow(np.transpose(stripalong_x[:,:,i])), plt.show()
        myft_peak_x += np.abs(np.fft.fftshift(np.fft.fft(np.sum(stripalong_x[:,:,i], 0))))
        #plt.plot(myft_peak_x), plt.show()
    
    # filter out zero order and negative spectrum
    myft_peak_x[0:int(cropsize[0]/2+10)]=0
    max_myft_peak_x = np.argmax(myft_peak_x);
    mygrating_const_x = 2*(cropsize[0]/2)/(max_myft_peak_x-cropsize[0]/2)
    print('mygrating_const_x: '+str(mygrating_const_x))
    #plt.plot(myft_peak_x)
        
    print('# Observe Peaks along Y')
    myft_peak_y = np.zeros((cropsize[0], ))
    for i in range(Niter):
        #plt.imshow(np.transpose(stripalong_y[:,:,i])), plt.show()
        myft_peak_y += np.abs(np.fft.fftshift(np.fft.fft(np.sum(stripalong_y[:,:,i], 1))))
        #plt.plot(myft_peak_y), plt.show()
    
    # filter out zero order and negative spectrum
    myft_peak_y[0:int(cropsize[0]/2+10)]=0
    max_myft_peak_y = np.argmax(myft_peak_y);
    mygrating_const_y = 2*(cropsize[0]/2)/(max_myft_peak_y-cropsize[0]/2)
    print('mygrating_const_y: '+str(mygrating_const_y))
    
    # generate grating
    unit_cell = np.zeros((int(np.ceil(mygrating_const_x)), int(np.ceil(mygrating_const_y))))
    unit_cell[0,0] = 1
    unit_cell_size = unit_cell.shape
    
    # generate final grating
    mygrid = np.repeat(unit_cell, int(np.ceil(cropsize[0]/unit_cell_size[0])+1), int(np.ceil(cropsize[1]/unit_cell_size[1]))+1)
    mygrid = utils.extract2D(mygrid, cropsize)
    
    return mygrid, mypeaks


def detect_and_warp_lattice_old(mygrid, mypeaks_crop, debug=False):
    #%% Idea: Detect the optical flow between the two frames and let the deltas be the origin of the de-distortion
    # Parameters for Farneback optical flow
    mypeaks_blurred = gaussian_filter(1.*mypeaks_crop, sigma=7)
    mygrid_masked = mypeaks_blurred*mygrid
    plt.imshow(mygrid_masked)
    
    # define the
    prevgray = np.uint16(mygrid_masked/np.max(mygrid_masked)*2**8)
    nextgray = np.uint16(mypeaks_crop/np.max(mypeaks_crop)*2**8-1)
    flow = cv2.calcOpticalFlowFarneback(prevgray, nextgray, None, pyr_scale = .5, levels=3, winsize=9, iterations=5, poly_n=9, poly_sigma=1.2, flags=0)
    
    myflow = utils.draw_flow(prevgray, flow)
    
    # map the two images see how we'Re doing 
    mygrating_mapped = utils.warp_flow(mygrid, flow)
    
    if(debug):
        plt.imshow(np.sum(myflow, 2)), plt.colorbar(), plt.show()


        # Draw the flow along X/Y
        plt.imshow(flow[:,:,0]), plt.colorbar(), plt.show()
        plt.imshow(flow[:,:,1]), plt.colorbar(), plt.show()
        
        plt.imshow(gaussian_filter(flow[:,:,0], sigma=7)), plt.colorbar(), plt.show()
        plt.imshow(gaussian_filter(flow[:,:,1], sigma=7)), plt.colorbar(), plt.show()
    
    return mygrating_mapped

def detect_warp_lattice(mygrid, myframe_crop, debug=False, blurkernel = 7):
    #%% Hmm..not working too great..
    mygrid_blurred = gaussian_filter(np.float32(mygrid), sigma=1.2) 
    plt.imshow(mygrid_blurred), plt.show()
    
    # Idea: Detect the optical flow between the two frames and let the deltas be the origin of the de-distortion
    # Parameters for Farneback optical flow
    prevgray = np.uint16(mygrid_blurred/np.max(mygrid_blurred)*2**8)
    gray = np.uint16(myframe_crop/np.max(myframe_crop)*2**8-1)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, pyr_scale = .5, levels=3, winsize=45, iterations=5, poly_n=9, poly_sigma=1.2, flags=0)

    # smoothen the flow to avoid noise
    flow_smooth = flow*0
    flow_smooth[:,:,0] = gaussian_filter(flow[:,:,0], sigma=blurkernel) 
    flow_smooth[:,:,1] = gaussian_filter(flow[:,:,1], sigma=blurkernel) 
    

    return flow_smooth

def run_warp_lattice(mygrid, myflow, debug=False):
    # map the two images see how we'Re doing 
    mygrating_mapped = utils.warp_flow(mygrid, myflow)

    if(debug):
        plt.imshow(np.sum(myflow, 2)), plt.colorbar(), plt.show()
        
        # Draw the flow along X/Y
        plt.imshow(myflow[:,:,0]), plt.colorbar(), plt.show()
        plt.imshow(myflow[:,:,1]), plt.colorbar(), plt.show()
    

        plt.imshow(np.float32(myframe_crop +(mygrating_mapped*np.max(myframe_crop)*.5)))    
        tif.imsave('result_flow_smooth.tif', np.float32(mygrid +(mygrating_mapped*np.max(myframe_crop)*.5)))
        plt.imsave('result_flow_smooth.png', np.float32(mygrid +(mygrating_mapped*np.max(myframe_crop)*.5)))
        plt.imsave('result_flow_smooth_mapx.png', mygrid[:,:,0])
        plt.imsave('result_flow_smooth_mapy.png', mygrid[:,:,1])
        
        plt.imshow(np.float32(myframe_crop +(mygrating_mapped*np.max(myframe_crop)*.5)))
        
    return mygrating_mapped




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


def get_shift_coordinates(my_videofile, scalesize, debug = False):
    '''
    my_videofile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/CheapConfocal/MATLAB/P12-HWP9_2018-11-1514.28.24.mp4.tif'
    scalesize = 1;
    debug = True
    get_shift_coordinates(my_videofile, scalesize)
    '''
    # 1.) Read all frames 
    framelist = []
    for i in range(10000):
        try:
            # read the first frame
            framelist.append(readtif(my_videofile, key=i, scalesize=scalesize))
            if(debug): print(str(i))
        except(IndexError):
            print('End of file is reached @ ' + str(i))
            break
            
        
    #%% 2.) project the time-stack along X and find the shift of the pupils using Hough Transforms
    # convert to array
    framelist = np.array(framelist)    
    
    # get X-projection    
    proj_x = np.std(framelist, axis=2)
    if(debug): plt.imshow(proj_x)

    # find Hough Transform and get their slope
    mymx = FindHoughLinesP(proj_x)
    
    #%% 3.) project the time-stack along Y and find the shift of the pupils using Hough Transforms
    # get Y-projection    
    proj_y = np.std(framelist, axis=1)
    proj_y = np.log(proj_y**7)
    myhighpassfilter = np.fft.fftshift(np.squeeze(rr(inputsize_x=proj_y.shape[0], inputsize_y=proj_y.shape[1], inputsize_z=1)>(np.min(proj_y.shape)*.1)))
    proj_y = np.real(np.fft.ifft2(np.fft.fft2(proj_y)*myhighpassfilter))
    if(debug): plt.imshow(myhighpassfilter),  plt.show()
    if(debug): plt.imshow(proj_y), plt.show()
    if(debug): plt.imshow(proj_y), plt.show()
    
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
        if(debug): plt.title('rotated projection along y'), plt.imshow(img), plt.show()
        #if(debug): plt.title('finite difference of prjection along y'), plt.plot(mygrad), plt.show()
        print('Gradl @ ' +str(iangle) + ' is '+str(np.mean(mygrad)) )

    # MAGIC
    proj_y_grad = np.std(proj_y, axis=1)
    plt.plot(proj_y_grad), plt.show()
    
    finite_diff_y = np.abs(proj_y[0:-2,:] - proj_y[1:-1,:])# -np.roll(proj_y, [1], axis=0))
    if(debug): plt.imshow(finite_diff_y), plt.show()
    # Still ugly to find the steps in Y-Shifts 
    mysteps = np.sum(finite_diff_y, axis=1)
    mysteps = np.squeeze(cv2.GaussianBlur(mysteps, (1, 15), 0))

    # detect peaks - each peak corresponds to one change of the illuminatino pattern in Y 
    peaks, _ = find_peaks(np.squeeze(mysteps), distance=20)
    if(debug): 
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


def fit_line(ydata, debug=False):
    from scipy.optimize import curve_fit
    def func(x, m, b):
        return m*x+b
    xdata = range(ydata.shape[0])
    popt_x, pcov_x = curve_fit(func, xdata, ydata)
        
    if(debug):
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
