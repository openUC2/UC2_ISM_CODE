# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(6, 4))
mpl.rc('image', cmap='gray')

from scipy.ndimage.filters import gaussian_filter
import numpy as np
import cv2
import tifffile as tif

import scipy.signal
from scipy.interpolate import interp1d
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
cropsize = ((200, 200))     # cropped size from the original image (around the center)


# OPRA parameters
mypinholesize = 5           # pixels of raw-frame holding the psf (even number only!)
psfradius = 3             # number of pixels forming the gaussian PSF
ismupscalefac = 2           # Super-resolution? Either 1 or 2
is_reducetime = True        # average frames along time

debug = False # show images?
isvideo = True # want to write the video?


#%% --------------------------------------------------------------------------
#
#--------------------------CODE STARTS HERE-----------------------------------
#
#%  --------------------------------------------------------------------------


#%%-------------------------- FILE I/O ---------------------------------------
print('Read Image Stack ... This may take a while!')
if(0):
    try:
        ismstack_raw = ism.readtifstack('./Data/'+my_videofile, scalesize)
    except:
        ismstack_raw = ism.readtifstack(my_videofile)

# filter out double-time shots
ismstack = ism.reducetime(ismstack_raw)


#%%-------------------------- Copute Super-Confocal --------------------------
print('First we want to produce a super-confocal image R. Heintzman et al 2006')
# Super=Max + Min-2*Avg
ismstack = np.float32(ismstack)
mysuperconfocal = np.max(ismstack, axis=0)+np.min(ismstack,axis=0)-2*np.mean(ismstack,axis=(0))
mybf =  np.mean(ismstack, axis=0) 

plt.figure()
plt.subplot(121), plt.title('Superconfocal'), plt.imshow(mysuperconfocal, cmap='hot'), plt.colorbar()
plt.subplot(122), plt.title('Brightfield'), plt.imshow(mybf, cmap='hot'), plt.colorbar(), plt.show()

#%%-------------------------- Peak-Detection ---------------------------------
print('Estimating the peaks from a single frame.')
Nimages = ismstack.shape[0]     # How many images you want to use for the ISM image? (fps~20, illumination~5)
mysize = ismstack.shape[1:2]    # size from the original image
myframeindex=55                 # some frame index which is used to estimate the grid
mindist = int(5)      # minimal distance between peaks 

# compute the peaks from the single frame
resize_fac = 2
test_frame = scipy.ndimage.zoom(np.squeeze(ismstack[myframeindex,:,:]), resize_fac, order=3)
mypeaks_map, mypeaks_pos = ism.get_peaks_from_image(test_frame,mindist=mindist*resize_fac ,debug=True)

#%% Fit lines to üeaks in vertical direction

# determine range  
polyn = 2 # order of ponlynimal fit 
max_std = 3*resize_fac # maximum of stdv of fit 
searchdist_vert = 3*resize_fac
searchdist_horz = 3*resize_fac

# fit lines to the peaks
my_fit_vert, my_linfit_vert, my_vert, my_fit_horz, my_linfit_horz, my_horz = ism.fit_illpattern_ism(mypeaks_pos, polyn = polyn, max_std = max_std, searchdist_vert = searchdist_vert, searchdist_horz = searchdist_horz, debug = False)

# generate the deformed illumination grid from the first frame
my_grid, my_grid_index  = ism.generate_illumination_grid(test_frame, my_fit_vert, my_vert, my_fit_horz, my_horz)

# generate the ideal illumination grid from the first frame
my_lingrid, my_lingrid_index  = ism.generate_illumination_grid(test_frame, my_linfit_vert, my_vert, my_linfit_horz, my_horz)

# visualize it
plt.subplot(121)
plt.title('Nonlinear fit')
plt.imshow((1+test_frame**1).T)
for i in range(0, my_fit_vert.shape[0]): plt.plot(my_fit_vert[i,:], my_vert)
for i in range(0, my_fit_horz.shape[0]): plt.plot(my_horz, my_fit_horz[i,:])
plt.plot(mypeaks_pos[:,0],mypeaks_pos[:,1],'x')


# visualize it
plt.subplot(122)
plt.title('Linear fit')
plt.imshow((1+test_frame**1).T)
for i in range(0, my_linfit_vert.shape[0]): plt.plot(my_linfit_vert[i,:], my_vert)
for i in range(0, my_linfit_horz.shape[0]): plt.plot(my_horz, my_linfit_horz[i,:])
plt.plot(mypeaks_pos[:,0],mypeaks_pos[:,1],'x')
plt.show()


plt.title('Extracted Coordinates from the ISM Frame')
plt.subplot(131),plt.imshow(gaussian_filter(1.*my_grid, sigma=1))
plt.subplot(132),plt.imshow(gaussian_filter(1.*my_lingrid, sigma=1))
plt.subplot(133),plt.imshow(gaussian_filter(1.*my_lingrid+my_grid, sigma=2)), plt.show()

''' now that we have the deformed and the "original" grid, we should compute the gradient 
(e.g. difference coordinate vector), from each ideal to each deformed point 
this roughly equals the camera calibration matrix 

Once we have that, we get the coordinate transformation matrix which maps 
each ideal spot to the "true" deformed one'''





    #%%

# Shift the lattice according to the detected optical flow
mygrid_mapped = ism.run_warp_lattice(mygrid, myflow)

# Extract all pinholes of the frame and place them on a new canvas with twice the size
mygrid_mapped_peak = utils.detect_peaks(mygrid_mapped)
if(debug): # Display for debugging purposes
    plt.imshow(myframe_crop +(mygrid_mapped*np.max(myframe_crop)*.5)), plt.colorbar(), plt.show()

# find shift between frames in advance
if(findshift_in_advance): myshifts = ism.find_shift_grid(ismstack,mygrid_raw,cropsize)


#%% Extract the pinholes according to the detected peaks

# generate a gaussian mask to pinhole the images
mypinhole = utils.generate_gaussian2D(mysize=mypinholesize, sigma=psfradius, mu=1.0)
mypinhole=mypinhole/np.max(mypinhole)
if(debug): plt.imshow(mypinhole), plt.colorbar(), plt.show()
ierror=0


#% we want to average over multiple frames shot at the same illumination pattern
ism_result = np.zeros((cropsize[0]*ismupscalefac, cropsize[1]*ismupscalefac))
ism_result_stack = np.zeros((Nimages, cropsize[0]*ismupscalefac, cropsize[1]*ismupscalefac))
grid_stack = np.zeros((Nimages, cropsize[0], cropsize[1]))
testimage = np.zeros((cropsize[0]*ismupscalefac,cropsize[1]*ismupscalefac))
testpinholes = np.zeros((cropsize[0]*ismupscalefac,cropsize[1]*ismupscalefac))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter(my_videofile+'result_frame_registration_ISM.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (cropsize[0],cropsize[1]))


#%% Here we place the pinholes in each frame and average by the number of detected pinholes per frame
for i_image in np.arange(0,Nimages-1):#range(0, 2, Nimages):
    # Here we only estimate the shift between the current frame and the best lattice

    # Read the frame and crop it
    current_frame = ismstack[i_image, :,:]
    current_frame = utils.extract2D(current_frame, cropsize) # first get ROI
    
    # find the shift 
    if(findshift_in_advance): 
        global_shift = myshifts[i_image,:]#utils.find_shift_lattice(mygrid_raw,current_frame)
    else:
        global_shift = utils.find_shift_lattice(mygrid_raw,current_frame)
        residual_shift = np.floor(global_shift) - global_shift
        global_shift = np.floor(global_shift)
        
    mygrid = np.zeros(mygrid_raw.shape)
    mygrid = np.roll(1*mygrid_raw, -int(global_shift[0]), axis=0)
    mygrid = np.roll(mygrid, -int(global_shift[1]), axis=1)
    grid_stack[i_image,:,:]=mygrid
    # Shift the lattice according to the detected optical flow
    mygrid_mapped = ism.run_warp_lattice(1.*mygrid, myflow*applyflow)

    # Extract all pinholes of the frame and place them on a new canvas with twice the size
    mygrid_mapped = utils.detect_peaks(mygrid_mapped)>.1
    
    # Find the pinhole positions
    myxcoord, myycoord = np.where(mygrid_mapped>0.1)
    
    # Number of detected pinholes
    n_pinholes = myxcoord.shape[0]

    # place each pinhole on a 2D grid and save it
    for i in range(n_pinholes):
        #testimage[myxcoord, myycoord] = 1
        try:

            # cut out the pinhole around the detected center
            masked_subset = utils.extract2D(current_frame, ((mypinholesize, mypinholesize)), center=((myxcoord[i], myycoord[i])))
            if(debug and np.mod(i, 1)==0): 
                plt.imshow(masked_subset), plt.colorbar(), plt.show()
            #plt.imshow(masked_subset), plt.show()
            # compute the positions for the cut-out pinholes
            offset_x = 0
            offset_y = 0
            new_index_x_1 = int(ismupscalefac*myxcoord[i]+offset_x-np.floor(mypinholesize/2))
            new_index_x_2 = int(ismupscalefac*myxcoord[i]+offset_x+np.floor(mypinholesize/2)+1)
            new_index_y_1 = int(ismupscalefac*myycoord[i]+offset_y-np.floor(mypinholesize/2))
            new_index_y_2 = int(ismupscalefac*myycoord[i]+offset_y+np.floor(mypinholesize/2)+1)
    
            # place the pinholes inside bigger frame
            M = np.float32([[1,0,residual_shift[0]],[0,1,residual_shift[1]]])
            pinhole_tmp = cv2.warpAffine(mypinhole,M,(mypinholesize,mypinholesize)) 
            masked_subset = masked_subset*pinhole_tmp
             
            ism_result[new_index_x_1:new_index_x_2, new_index_y_1:new_index_y_2] += masked_subset
            ism_result_stack[i_image,new_index_x_1:new_index_x_2, new_index_y_1:new_index_y_2]= masked_subset
            testpinholes[new_index_x_1:new_index_x_2, new_index_y_1:new_index_y_2] += pinhole_tmp
            
       
        except(ValueError):
            #print('Error @ ' + str(i))
            ierror = ierror+1#print('Extracted ROI not at the edge')
    
    #plt.imshow(current_frame +(mygrid*np.max(current_frame)*.5)), plt.colorbar(), plt.show()
    # Write the frame into the file 'output.avi'
   # Display for debugging purposes
    if(isvideo):        
        video_frame = (current_frame +(mygrid_mapped*np.max(current_frame)*.5))
        video_frame = np.uint8(video_frame/np.max(video_frame)*(2**8-1))
        video_frame = np.transpose(np.stack([video_frame, video_frame, video_frame]), (1, 2, 0))
        out.write(video_frame)

out.release() 

# write second movie
out = cv2.VideoWriter(my_videofile+'result_over_time_ISM.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 5, (ismupscalefac*cropsize[0],ismupscalefac*cropsize[1]))
for i in range(Nimages): 
    video_frame = ism_result_stack[i,:,:]
    video_frame = np.uint8(video_frame/np.max(video_frame)*(2**8-1))
    video_frame = np.transpose(np.stack([video_frame, video_frame, video_frame]), (1, 2, 0))
    out.write(video_frame)
out.release() 

#% Display results
plt.title('Result'), plt.imshow(ism_result), plt.show()
  
# find overlap
#for i in range(1,15):
#    plt.title('Projection of timesteps along t: '+str(i)), plt.imshow(np.sum(ism_result_stack[0:-i,:,:], axis=0)),plt.show()
# normalize intensity along time - well..not nice, but this is due to the projectors aliasing..
lastslice=12
plt.title('Projection of timesteps along t'), plt.imshow(np.sum(ism_result_stack[100:-lastslice,:,:], axis=0)),plt.show()
plt.title('Projection of timesteps along x'), plt.imshow(np.sum(ism_result_stack, axis=1)),plt.show()
plt.title('Projection of timesteps along y'), plt .imshow(np.sum(ism_result_stack, axis=2)),plt.show()

# normalize intensity along time - well..not nice, but this is due to the projectors aliasing..
plt.title('Projection of timesteps along t'), plt.imshow(np.sum(grid_stack, axis=0)),plt.show()
plt.title('Projection of timesteps along x'), plt.imshow(np.sum(grid_stack, axis=1)),plt.show()
plt.title('Projection of timesteps along y'), plt.imshow(np.sum(grid_stack, axis=2)),plt.show()


#%%
from scipy import ndimage
#midealspots = ndimage.convolve(midealspots, mypinhole, mode='constant', cval=0.0)
ismsum = np.sum(ism_result_stack, 0)
myfinalresult = (ismsum/np.max(ismsum))/(testpinholes/np.max(testpinholes))#.25*gaussian_filter(testpinholes,0.7)#testpinholes
myfinalresult=np.nan_to_num(myfinalresult)

plt.title('Final Result'), plt.imshow(myfinalresult), plt.show()

plt.imsave('publi_finalreslt.png', myfinalresult)

import tifffile as tif
tif.imsave('publi_finalreslt.tif', myfinalresult)    

rainerresult = (np.max(ismstack,(0))+np.min(ismstack,(0)))-(2*np.mean(ismstack,(0)))
plt.imsave('publi_mx+min-2avg.png',utils.extract2D(rainerresult, cropsize))
plt.imsave('publi_mybf.png',utils.extract2D(mybf, cropsize))


plt.figure()
myframe_crop = utils.extract2D(ismstack[myframeindex,:,:,], cropsize) # first get ROI

plt.subplot(131), plt.title('Brightfield'), plt.imshow(utils.extract2D(np.sum(ismstack,axis=0), cropsize), cmap='hot')#, plt.colorbar()
plt.subplot(132), plt.title('Superconfocal'), plt.imshow(utils.extract2D(mysuperconfocal, cropsize), cmap='hot')#, plt.colorbar()
plt.subplot(133), plt.title('ISM Result'), plt.imshow(myfinalresult, cmap='hot')#, plt.colorbar()
plt.savefig('Result_ISM_all.png')
plt.show()

import scipy.io as sio
sio.savemat('np_exchange.mat', {'mypeaks_pos':mypeaks_pos})

