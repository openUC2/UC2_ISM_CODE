# -*- coding: utf-8 -*-
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import os
import tifffile as tif
import scipy.signal

import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline # Optionally, tweak styles.
mpl.rc('figure',  figsize=(12, 10))
mpl.rc('image', cmap='gray')

# own tools
import utils as utils
import ismtools2 as ism
import NanoImagingPack as nip
#%% --------------------------------------------------------------------------
#
#--------------------------Define Parameters----------------------------------
#
#%  --------------------------------------------------------------------------
# Define input files
    
# video GFP cells 
my_videofile = '/Users/Bene/Downloads/2019-09-17 10.09.27_substack_bin2.tif'
my_videofile = './Data/2019-09-17 10.09.27.mp4_bin2_193_end_756-1256.tif'
is_debug = False

# OPRA parameters
mypinholesize = 9           # pixels of raw-frame holding the psf (even number only!)
psfradius = 3             # number of pixels forming the gaussian PSF
ismupscalefac = 2           # Super-resolution? Either 1 or 2
is_reducetime = True        # average frames along time

debug = False # show images?
is_video = False # want to write the video?


#%% --------------------------------------------------------------------------
#
#--------------------------CODE STARTS HERE-----------------------------------
#
#%  --------------------------------------------------------------------------


#%%-------------------------- FILE I/O ---------------------------------------
print('Read Image Stack ... This may take a while!')
if(1):
   ismstack_raw = ism.readtifstack(my_videofile)

# filter out double-time shots
if(0):
    ismstack = ism.reducetime(ismstack_raw)
else:
    ismstack = (ismstack_raw)
tif.imsave(os.path.split(my_videofile)[-1]+'_reducedtime.tif',ismstack)

#%%-------------------------- Copute Super-Confocal --------------------------
print('First we want to produce a super-confocal image R. Heintzman et al 2006')
# Super=Max + Min-2*Avg

ismstack = np.float32(ismstack)
mysuperconfocal = np.max(ismstack, axis=0)+np.min(ismstack,axis=0)-2*np.mean(ismstack,axis=(0))
mybf =  np.mean(ismstack, axis=0) 

plt.figure()
plt.subplot(121), plt.title('Superconfocal'), plt.imshow(mysuperconfocal, cmap='hot'), plt.colorbar()
plt.subplot(122), plt.title('Brightfield'), plt.imshow(mybf, cmap='hot'), plt.colorbar(), plt.show()

#my_grid_old = my_grid*0
#my_lingrid_old = my_grid*0
#%%-------------------------- Peak-Detection ---------------------------------
print('Estimating the peaks from a single frame.')
resize_fac = 2
Nimages = ismstack.shape[0]     # How many images you want to use for the ISM image? (fps~20, illumination~5)
mysize = ismstack.shape[1]    # size from the original 

my_grid_old = np.zeros((mysize*resize_fac,mysize*resize_fac))
my_lingrid_old = np.zeros((mysize*resize_fac,mysize*resize_fac)) 

# safe the indeces from detected peaks in a list 
my_all_index_list = []
my_all_linindex_list = []

for myframeindex in range(1,Nimages): # some frame index which is used to estimate the grid
    mindist = int(5)      # minimal distance between peaks 
    
    # compute the peaks from the single frame

    test_frame = scipy.ndimage.zoom(np.squeeze(ismstack[myframeindex,:,:]), resize_fac, order=3)
    mypeaks_map, mypeaks_pos = ism.get_peaks_from_image(test_frame,mindist=mindist*resize_fac ,debug=is_debug)
    
    #% Fit lines to üeaks in vertical direction
    
    # determine range  
    polyn = 2 # order of ponlynimal fit 
    max_std = 3*resize_fac # maximum of stdv of fit 
    searchdist_vert = 3*resize_fac
    searchdist_horz = 3*resize_fac
    
    # fit lines to the peaks
    my_fit_vert, my_linfit_vert, my_vert, my_fit_horz, my_linfit_horz, my_horz = ism.fit_illpattern_ism(mypeaks_pos, polyn = polyn, max_std = max_std, searchdist_vert = searchdist_vert, searchdist_horz = searchdist_horz, debug = is_debug)

    # generate the deformed illumination grid from the first frame - array => 0-index is y,, 1-index is x
    my_grid, my_grid_index  = ism.generate_illumination_grid(test_frame, my_fit_vert, my_vert, my_fit_horz, my_horz)
    
    # generate the ideal illumination grid from the first frame - array => 0-index is y,, 1-index is x
    my_lingrid, my_lingrid_index  = ism.generate_illumination_grid(test_frame, my_linfit_vert, my_vert, my_linfit_horz, my_horz)
    
    my_grid_old += my_grid
    my_lingrid_old += my_lingrid
    
    # save all indices for later use
    my_all_linindex_list.append(np.array(my_lingrid_index))
    my_all_index_list.append(np.array(my_grid_index))
    
    if(is_debug):
        # visualize it
        plt.subplot(121)
        plt.title('Nonlinear fit')
        plt.imshow((1+test_frame)**.2)
        for i in range(0, my_linfit_vert.shape[0]): plt.plot(my_vert, my_fit_vert[i,:])
        for i in range(0, my_linfit_horz.shape[0]): plt.plot(my_fit_horz[i,:], my_horz)
        plt.plot(mypeaks_pos[:,1],mypeaks_pos[:,0],'x')
        plt.plot(np.array(my_lingrid_index)[1,:],np.array(my_lingrid_index)[0,:],'o')        
        
        
        # visualize it
        plt.subplot(122)
        plt.title('Linear fit')
        plt.imshow((1+test_frame)**.2)
        for i in range(0, my_linfit_vert.shape[0]): plt.plot(my_vert, my_linfit_vert[i,:])
        for i in range(0, my_linfit_horz.shape[0]): plt.plot(my_linfit_horz[i,:], my_horz)
        plt.plot(mypeaks_pos[:,1],mypeaks_pos[:,0],'x')
        plt.plot(np.array(my_grid_index)[1,:],np.array(my_grid_index)[0,:],'o')        
        plt.savefig('myfig'+str(myframeindex)+'.png')
        plt.show()
    
    if(is_debug):
        plt.title('Extracted Coordinates from the ISM Frame')
        plt.subplot(131),plt.imshow(gaussian_filter(1.*my_grid, sigma=1))
        plt.subplot(132),plt.imshow(gaussian_filter(1.*my_lingrid, sigma=1))
        plt.subplot(133),plt.imshow(gaussian_filter(1.*my_lingrid+my_grid, sigma=2)), plt.show()
    
    ''' now that we have the deformed and the "original" grid, we should compute the gradient 
    (e.g. difference coordinate vector), from each ideal to each deformed point 
    this roughly equals the camera calibration matrix 
    
    Once we have that, we get the coordinate transformation matrix which maps 
    each ideal spot to the "true" deformed one'''


# All peaks are now contained in the following list: my_all_index_list

plt.title('SUM pinholes')
plt.subplot(121),plt.imshow(gaussian_filter(1.*my_grid_old, sigma=1))
plt.subplot(122),plt.imshow(gaussian_filter(1.*my_lingrid_old, sigma=1)), plt.show()

#%%
'''
#% Extract the pinholes according to the detected peaks
'''

# generate a gaussian mask to pinhole the images
mypinhole = utils.generate_gaussian2D(mysize=mypinholesize, sigma=mypinholesize/1, mu=1.0) # np.ones((mypinholesize, mypinholesize))# 
mypinhole=mypinhole/np.max(mypinhole) # scale it to 1 - max, makes sense? Just linear scaling factor?! 
plt.title('Pinhole'), plt.imshow(mypinhole), plt.colorbar(), plt.show()



#% we want to average over multiple frames shot at the same illumination pattern
ism_result = np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))
ism_result_stack = np.zeros((Nimages, mysize*ismupscalefac, mysize*ismupscalefac))
grid_stack = np.zeros((Nimages, mysize, mysize))
testimage = np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))
all_sum_pinholes = np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))



#% Here we place the pinholes in each frame and average by the number of detected pinholes per frame
for i_image in np.arange(0,Nimages-1):#range(0, 2, Nimages):
    # Here we only estimate the shift between the current frame and the best lattice

    # Read the frame and crop it
    current_frame = ismstack[i_image, :,:]

    # Number of detected pinholes
    n_pinholes = my_all_index_list[i_image].shape[1]

    # Get the coordanates per frame
    myxcoord = my_all_index_list[i_image][1]
    myycoord = my_all_index_list[i_image][0]

    current_frame = nip.resample(current_frame, (ismupscalefac, ismupscalefac))
    #%
    # place each pinhole on a 2D grid and save it
    for i_pinhole in range(n_pinholes):
        # cut out the pinhole around the detected center
        my_centerpos_i_y, my_centerpos_i_x = myycoord[i_pinhole], myxcoord[i_pinhole]
        masked_subset = nip.extract(current_frame, ROIsize = (mypinholesize, mypinholesize), centerpos = (my_centerpos_i_y, my_centerpos_i_x))
        
        # display if necessary
        if(is_debug and np.mod(i, 1)==0): 
            plt.imshow(masked_subset), plt.colorbar(), plt.show()
            print(str(new_index_x_1) + '/' + str(new_index_y_1))

        # compute the positions for the cut-out pinholes
        new_index_x_1 = int(my_centerpos_i_x-mypinholesize//2)
        new_index_x_2 = int(my_centerpos_i_x+mypinholesize//2+1)
        new_index_y_1 = int(my_centerpos_i_y-mypinholesize//2)
        new_index_y_2 = int(my_centerpos_i_y+mypinholesize//2+1)


        # apply the computational pinhole to the intensity data to block out-of focus light
        masked_subset = masked_subset*mypinhole
         
        # place the pinholes inside bigger frame
        ism_result[new_index_y_1:new_index_y_2, new_index_x_1:new_index_x_2] += masked_subset
        ism_result_stack[i_image,new_index_y_1:new_index_y_2, new_index_x_1:new_index_x_2] += masked_subset
        all_sum_pinholes[new_index_y_1:new_index_y_2, new_index_x_1:new_index_x_2] += mypinhole
        
    print('processing Frame No: '+str(i_image) + ' of ' + str(Nimages-1))
       
#%    
ism_result_mod = (ism_result/np.max(ism_result))/(all_sum_pinholes/np.max(all_sum_pinholes))#.25*gaussian_filter(testpinholes,0.7)#testpinholes
ism_result_mod[np.isnan(ism_result_mod)]=0
ism_result_final = ism_result-all_sum_pinholes

plt.figure()
plt.subplot(131), plt.title('Superconfocal'), plt.imshow(mysuperconfocal, cmap='hot')#, plt.colorbar()
plt.subplot(132), plt.title('Brightfield'), plt.imshow(mybf, cmap='hot')#, plt.colorbar()
#plt.subplot(223), plt.title('ISM'), plt.imshow(ism_result_final, cmap='hot')#, plt.colorbar(), plt.show()
plt.subplot(133), plt.title('ISM (mod)'), plt.imshow(ism_result_mod, cmap='hot')#, plt.colorbar(), plt.show()




#%% 
''' save the images'''
tif.imsave(os.path.split(my_videofile)[-1]+'_Superconfocal.tif', np.float32(mysuperconfocal))
tif.imsave(os.path.split(my_videofile)[-1]+'_Brightfield.tif', np.float32(mybf))
tif.imsave(os.path.split(my_videofile)[-1]+'ISM_result_filtered.tif', np.float32(ism_result_mod))
