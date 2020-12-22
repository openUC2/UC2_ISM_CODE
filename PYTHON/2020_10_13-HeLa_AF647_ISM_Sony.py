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
mpl.rc('figure',  figsize=(20, 16))
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
# Define input file
myfolder = './DATA/'
myscalesize = .5 # downsample data - it's very likely that it's oversampled by 2x
myvideofile = '23.10.2020_800x600_HeLa_SiR_newSample_3.tif'
mydarkfile = "Alvium_Dark_500ms_gain11.tif"
mybackground = 0

# OPRA parameters
mypinholesize = 15          # pixels of raw-frame holding the psf (even number only!)
mypsfradius = 2             # number of pixels forming the gaussian PSF
ismupscalefac = 2           # Super-resolution? Either 1 or 2
is_reducetime = False        # average frames along time -does not work for now!?

# pattern search parameters
# determine range  
polyn = 1 # order of ponlynimal fit 
max_std = 3 # maximum of stdv of fit (in terms of "straightness")
# in RAW data pixels!
searchdist_vert = 8 # minimum seperation between two different lines (vertically)
searchdist_horz = searchdist_vert # minimum seperation between two different lines (horizontally)
mindist = np.min((searchdist_vert,searchdist_horz) )    # minimal distance between peaks 
  
# Switches
is_debug = False         # want to display/save all intermediate results?
is_fillmissing = True   # fill wholes in the grid?
is_usepeakfitonly = False
cropsize = None
is_denoise = False 
is_use_nearest = True # For all localized emitters replace grid coordinates with nearest peak?
is_findduplicates = False
#%% --------------------------------------------------------------------------
#
#--------------------------CODE STARTS HERE-----------------------------------
#
#%  --------------------------------------------------------------------------

if not mypinholesize%2: print("pinholesize must be an odd number! "), Error


rawfile = myfolder + myvideofile
if mydarkfile is not None: 
    darkfile = myfolder + mydarkfile
else: 
    darkfile = None
#%%-------------------------- FILE I/O ---------------------------------------
ismstack = utils.readismsack(rawfile, darkfile=None, backgroundval=0, myupscale=myscalesize, is_reducetime=is_reducetime, is_denoise=is_denoise, cropsize=cropsize)
#tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'_reducedtime.tif',ismstack)
Nimages = ismstack.shape[0]     # How many images you want to use for the ISM image? (fps~20, illumination~5)
mysize = ismstack.shape[1]    # size from the original 


#%%-------------------------- Copute Super-Confocal --------------------------
mysuperconfocal = ism.compute_superconfocal(ismstack, is_debug)
mybf = ism.compute_brightfield(ismstack, is_debug)


#%%-------------------------- Peak-Detection ---------------------------------
myg, myrot = ism.estimate_ism_pars(testframe=ismstack[2,:,:], mindist=mindist, radius_ft=0.02, is_debug=is_debug)
mindist = myg
rottheta = myrot
my_g_vert = my_g_horz = myg

#%%-------------------------- Allocate Memory ---------------------------------

# save all grids 
my_all_grid = []
my_all_peaks = []

# allocate memory for the grid
my_grid_old = np.zeros((mysize,mysize))
my_lingrid_old = np.zeros((mysize,mysize)) 

# safe the indeces from detected peaks in a list 
my_all_index_list = []
my_all_linindex_list = []
my_g_vert_list = []
my_g_horz_list = []

#%%
# iterate over all frames and detect the peaks
for myframeindex in range(0,Nimages): # some frame index which is used to estimate the grid

    #% compute peaks from frame
    # compute the peaks from the single frame - upscale it to have sub-pixel accuracy?
    test_frame = np.squeeze(ismstack[myframeindex,:,:])                                     # select frame
    if rottheta!= 0: test_frame = nip.rot2d(test_frame,rottheta,padding=False)#; if(is_debug): print("Rotating frame")              # rotate if necessary
    mypeaks_map, mypeaks_pos = ism.get_peaks_from_image(test_frame,mindist=mindist, is_debug=is_debug)
    my_all_peaks.append(mypeaks_pos.T)                                                      # save frames for later
    
    #% fit the grating to the peak positions
    print('Computing frame '+str(myframeindex)+'/'+str(Nimages))
    if is_usepeakfitonly:
        # This is if you only want to use the detected peaks for Pinhole selection
        # save all indices for later use     
        my_all_index_list.append(mypeaks_pos.T)
    else:
        # do you want to fit a proper grating to the detected peaks?
        my_g_vert_list.append(my_g_vert)
        my_g_horz_list.append(my_g_horz)      

        # There is a lot of magic going on. A set of lines is fit to the points in X and Y
        my_fit_vert, my_vert, my_fit_horz, my_horz, my_g_vert, my_g_horz = ism.fit_illpattern_ism(
                                    mypeaks_pos, polyn = polyn, 
                                    max_std = max_std, searchdist_vert = searchdist_vert, 
                                    searchdist_horz = searchdist_horz, 
                                    is_debug = is_debug, is_fillmissing=is_fillmissing, 
                                    g_vert=my_g_vert_list, g_horz=my_g_horz_list)
        
        
        # correct for mismatching grating period along horizontal direction
        # enforce global grating constant to match the fitted one..
        my_fit_horz = ism.adjust_grating_constant(my_fit_horz, 0, is_debug)
        my_fit_vert = ism.adjust_grating_constant(my_fit_vert, 0, is_debug)
        
        # generate the deformed illumination grid from the first frame - array => 0-index is y,, 1-index is x
        my_grid, my_grid_index  = ism.generate_illumination_grid(test_frame, my_fit_vert, my_vert, my_fit_horz, my_horz, is_debug)
        
    
        if is_use_nearest:
            try:
                # find shift between groundtruth peak (peakfit) and grid fit
                from scipy import spatial        
                tree = spatial.KDTree(np.vstack((my_grid_index)).T)
                mydist, myindex = tree.query(mypeaks_pos)
                
                # REPLACE WITH LOCALIZATION?!
                my_grid_index[0][myindex] = mypeaks_pos[:,0]
                my_grid_index[1][myindex] = mypeaks_pos[:,1]
                
                if(is_debug):
                    plt.plot(np.vstack(my_grid_index)[0,],np.vstack(my_grid_index)[1,],'o')
                    plt.plot(np.vstack(my_grid_index)[0,myindex],np.vstack(my_grid_index)[1,myindex],'o')
                    plt.plot(mypeaks_pos[:,0],mypeaks_pos[:,1],'x')
            except:
                print("Error with use nearest..")
        # append grid to the already exisiting grid points    
        my_grid_old += my_grid
    
        # save all indices for later use
        my_all_index_list.append(np.array(my_grid_index))
        my_all_grid.append(my_grid)
        
        if(is_debug):
            # visualize it
            plt.title('(Non)-Linear fit')
            plt.imshow((1+test_frame)**.2)
            for i in range(0, my_fit_vert.shape[0]): plt.plot(my_vert, my_fit_vert[i,:])
            for i in range(0, my_fit_horz.shape[0]): plt.plot(my_fit_horz[i,:], my_horz)
            plt.plot(mypeaks_pos[:,1],mypeaks_pos[:,0],'x')
            plt.plot(np.array(my_grid_index)[1,:],np.array(my_grid_index)[0,:],'o')
            plt.savefig(myfolder + 'myfig'+str(myframeindex)+'.png')
            plt.show()

        

#%% Reassignment of the pixels

''' now that we have the deformed and the "original" grid, we should compute the gradient 
(e.g. difference coordinate vector), from each ideal to each deformed point 
this roughly equals the camera calibration matrix 

Once we have that, we get the coordinate transformation matrix which maps 
each ideal spot to the "true" deformed one'''

if is_findduplicates:
    # find all duplicates 
    my_all_indices = my_all_index_list[0]
    for iiter in range(1,len(my_all_index_list)): 
        my_all_indices = np.hstack((my_all_indices, my_all_index_list[iiter]))
    
    
    # find duplicate entries of pinhole positions
    a = list(np.sqrt(my_all_indices[0,]**2 + my_all_indices[1,]**2))
    seen = set()
    uniq = [x for x in a if x not in seen and not seen.add(x)] 
    nonuniq = [x for x in a if x in seen] 
    
    from iteration_utilities import duplicates
    list(duplicates(a))

# All peaks are now contained in the following list: my_all_index_list

plt.title('SUM pinholes')
plt.subplot(121),plt.imshow(gaussian_filter(1.*my_grid_old, sigma=1))
plt.subplot(122),plt.imshow(gaussian_filter(1.*my_lingrid_old, sigma=1)), plt.show()

#%%
'''
#% Extract the pinholes according to the detected peaks
'''
mypinholesize = 11
mypsfradius = 2
ismupscalefac = 1

# generate a gaussian mask to pinhole the images
mypinhole = utils.generate_gaussian2D(mysize=mypinholesize, sigma=mypsfradius, mu=1.0) # np.ones((mypinholesize, mypinholesize))# 
#mypinhole=mypinhole/np.max(mypinhole) # scale it to 1 - max, makes sense? Just linear scaling factor?! 
#mypinhole=nip.rr(mypinhole.shape)<mypsfradius
plt.title('Pinhole'), plt.imshow(mypinhole), plt.colorbar(), plt.show()

#% we want to average over multiple frames shot at the same illumination pattern
ism_result = np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))
ism_result_stack = np.zeros((Nimages, mysize*ismupscalefac, mysize*ismupscalefac))
grid_stack = np.zeros((Nimages, mysize, mysize))
testimage = np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))
all_sum_pinholes = np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))
processed =  np.zeros((mysize*ismupscalefac, mysize*ismupscalefac))
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

    #current_frame = nip.resample(current_frame, (ismupscalefac, ismupscalefac))
    #%
    # place each pinhole on a 2D grid and save it
    for i_pinhole in range(n_pinholes):
        # cut out the pinhole around the detected center
        my_centerpos_i_y, my_centerpos_i_x = myycoord[i_pinhole], myxcoord[i_pinhole]
        masked_subset = nip.extract(current_frame, ROIsize = (mypinholesize, mypinholesize), centerpos = (my_centerpos_i_y-1, my_centerpos_i_x-1))

        # compute the positions for the cut-out pinholes
        new_index_x_1 = int(ismupscalefac*my_centerpos_i_x-mypinholesize//2)
        new_index_x_2 = int(ismupscalefac*my_centerpos_i_x+mypinholesize//2+1)
        new_index_y_1 = int(ismupscalefac*my_centerpos_i_y-mypinholesize//2)
        new_index_y_2 = int(ismupscalefac*my_centerpos_i_y+mypinholesize//2+1)

        # display if necessary
        if(is_debug and np.mod(i_pinhole, 1)==0): 
            plt.imshow(masked_subset), plt.colorbar(), plt.show()
            print(str(new_index_x_1) + '/' + str(new_index_y_1))


        # apply the computational pinhole to the intensity data to block out-of focus light
        masked_subset = masked_subset*mypinhole
        
        #make sure you only sum if there is no information placed from previous measurements
        try:
            if not processed[ismupscalefac*my_centerpos_i_x, ismupscalefac*my_centerpos_i_y]: #MANDATORY STEP
                # place the pinholes inside bigger frame
                ism_result[new_index_y_1:new_index_y_2, new_index_x_1:new_index_x_2] += masked_subset
                ism_result_stack[i_image,new_index_y_1:new_index_y_2, new_index_x_1:new_index_x_2] += masked_subset
                all_sum_pinholes[new_index_y_1:new_index_y_2, new_index_x_1:new_index_x_2] += mypinhole
                processed[ismupscalefac*my_centerpos_i_x, ismupscalefac*my_centerpos_i_y] += 1
    
        except:
            print("IndexError in ISM reconstruction...")
           
  
    print('processing Frame No: '+str(i_image) + ' of ' + str(Nimages-1))
       
#%    
print('SOMETHING IS WRONG WITH THE BACKGROUND ESTIMATEION!')
ism_result_mod = (ism_result/np.max(ism_result))/(all_sum_pinholes/np.max(nip.gaussf(all_sum_pinholes,0)))#.25*gaussian_filter(testpinholes,0.7)#testpinholes
ism_result_mod[np.isnan(ism_result_mod)]=0
ism_result_final = ism_result-all_sum_pinholes


# Show the final image
plt.figure()
cmap='hot'
plt.subplot(221), plt.title('Superconfocal'), plt.imshow(mysuperconfocal, cmap=cmap)#, plt.colorbar()
plt.subplot(222), plt.title('Brightfield'), plt.imshow(mybf, cmap=cmap)#, plt.colorbar()
plt.subplot(223), plt.title('ISM (mod)'), plt.imshow(ism_result_mod, cmap=cmap, vmin=0)#, plt.colorbar()#, plt.show()
plt.subplot(224), plt.title('ISM (result)'), plt.imshow(ism_result, cmap=cmap, vmin=0)#, plt.colorbar()#, plt.show()
plt.savefig(myfolder + os.path.split(myvideofile)[-1]+'_Subplot.png')

#%
''' save the images'''
tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'_Superconfocal.tif', np.float32(mysuperconfocal))
tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'_Brightfield.tif', np.float32(mybf))
tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'ISM_result_filtered.tif', np.float32(ism_result_mod))
tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'ISM_result.tif', np.float32(ism_result))
tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'ISM_result_filtered_stack.tif', np.float32(ism_result_stack))
tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'ISM_detectedpinholes_stack.tif', np.float32(np.array(my_all_grid)))

np.array(my_all_grid)