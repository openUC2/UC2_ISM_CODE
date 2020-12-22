# -*- coding: utf-8 -*-
import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
# %matplotlib inline

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(6, 4))
mpl.rc('image', cmap='gray')

import numpy as np
import cv2
import tifffile as tif

import scipy.signal

# own tools
import utils as utils
import ismtools as ism

#%% --------------------------------------------------------------------------
#
#--------------------------Define Parameters----------------------------------
#
#%  --------------------------------------------------------------------------
# Define input files
my_videofile = '/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/UC2/02-CheapConfocal/MATLAB/P12-HWP9_2018-11-1514.28.24.mp4.tif'; # the way to go! upscaling 1.06 
scalesize = 1.06 # scaling factor to get rid of the MP4 compression artifact?!
if(0):
    my_videofile = '2018-11-13 14.21.23.mp4.tif'
    my_videofile = '2018-11-13 16.25.10.mp4.tif'
    my_videofile = '2018-11-13 16.37.41.mp4.tif' # not working, 1.07
    my_videofile = '2018-11-14 14.37.42.mp4.tif' # too sparse! 1.07
    my_videofile = '2018-11-14 14.41.15.mp4_crop.tif' # not working
    my_videofile = '2018-11-14 15.22.30.mp4.tif' # 1.055 looks promising - maybe no optical flow!
    my_videofile = '2018-11-14 15.49.55.mp4.tif'
    my_videofile = 'P4_expo200ms_Python.tif'
    scalesize = 1.50 
    
# video GFP cells 
my_videofile = '/Users/Bene/Downloads/2019-09-17 10.09.27_substack_bin2.tif'
scalesize = 0.95 # scaling factor to get rid of the MP4 compression artifact?!
mypinholesize = 5           # pixels of raw-frame holding the psf (even number only!)
psfradius = 3             # number of pixels forming the gaussian PSF

#my_videofile = 'Haselpollen-groß-1-Huawei.mp4.tif';
#my_videofile = '2018-11-15 14.33.08.mp4.tif'


applyflow = 0 # 1 or 0
debug = False # show images?
isvideo = True # want to write the video?
# read the first frame

# Define the sizes
cropsize = ((200, 200))     # cropped size from the original image (around the center)



ismupscalefac = 2           # Super-resolution? Either 1 or 2
findshift_in_advance=False
is_findperfectlattice=True  # If true, a lattice with discrete frequencies is fitted to the Grid
#%% --------------------------------------------------------------------------
#
#--------------------------CODE STARTS HERE-----------------------------------
#
#%  --------------------------------------------------------------------------
# try to get a backgroundframe
if(True):
    try:
        ismstack_raw = ism.readtifstack('./Data/'+my_videofile, scalesize)
    except:
        ismstack_raw = ism.readtifstack(my_videofile, scalesize)

    # filter out double-time shots
    if(0):
        ismstack = ism.reducetime(ismstack_raw)
    else:
        ismstack = ismstack_raw
    print('Final number of Timesteps: '+str(ismstack.shape[0]))
    
#%%
print('First we want to produce a super-confocal image R. Heintzman et al 2006')
# Super=Max + Min-2*Avg
ismstack = np.float32(ismstack)
superconfocal = np.max(ismstack, axis=0)+np.min(ismstack,axis=0)-2*np.mean(ismstack,axis=(0))
plt.figure()
plt.subplot(121), plt.title('Superconfocal'), plt.imshow(superconfocal, cmap='hot'), plt.colorbar()
plt.subplot(122), plt.title('Brightfield'), plt.imshow(np.sum(ismstack,axis=0), cmap='hot'), plt.colorbar(), plt.show()


Nimages = ismstack.shape[0]              # How many images you want to use for the ISM image? (fps~20, illumination~5)
mysize = ismstack.shape[1:2]  # size from the original image
mybf =  np.mean(ismstack, axis=0)
plt.imshow(mybf), plt.colorbar(), plt.title('Brightfield equivalent'), plt.show()

''' Get the peaks of the illumination source '''
myframeindex=50
mymindistpattern = int(scalesize*5)
print('ATTENTION: check the distance here!!')
mypeaks_raw = ism.get_peaks_from_image(scipy.ndimage.filters.gaussian_filter(1.*ismstack[myframeindex,:,:,],1),mindist=mymindistpattern)
plt.imshow(np.float32(ismstack[myframeindex,:,:,])/np.float32(np.max(ismstack[myframeindex,:,:,])) + mypeaks_raw), plt.show()

# this gets the "perfect" lattice which represents the illumination positions of the laser
if(is_findperfectlattice):
    mygrid_raw, mypeaks_crop = ism.get_perfect_lattice(mypeaks_raw, cropsize)
else:
    mygrid_raw = utils.extract2D(mypeaks_raw, cropsize)
    mypeaks_crop = utils.extract2D(mypeaks_raw, cropsize)

# compare to real frame 
myframe_crop = utils.extract2D(ismstack[myframeindex,:,:,], cropsize) # first get ROI
plt.title('This is the "perfect" fitted grating (only grating constant)')
plt.imshow(myframe_crop +(mygrid_raw*np.max(myframe_crop )*.5))
tif.imsave('result.tif', np.float32(myframe_crop +(mygrid_raw*np.max(myframe_crop )*.5)))

asdf
# align the perfect lattice to the first frame
global_shift = utils.find_shift_lattice(mygrid_raw,myframe_crop)
mygrid = np.zeros(mygrid_raw.shape)
mygrid = np.roll(1*mygrid_raw, -int(global_shift[0]), axis=0)
mygrid = np.roll(mygrid, -int(global_shift[1]), axis=1)
if(debug): # Display for debugging purposes
    plt.imshow(myframe_crop +(mygrid*np.max(myframe_crop)*.5)), plt.colorbar(), plt.show()

# find lens distortion using optical flow
myflow = ism.detect_warp_lattice(mygrid, myframe_crop, debug=False, blurkernel = 11)

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
plt.subplot(132), plt.title('Superconfocal'), plt.imshow(utils.extract2D(superconfocal, cropsize), cmap='hot')#, plt.colorbar()
plt.subplot(133), plt.title('ISM Result'), plt.imshow(myfinalresult, cmap='hot')#, plt.colorbar()
plt.savefig('Result_ISM_all.png')
plt.show()
