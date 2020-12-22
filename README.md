# UC2 ISM algorithm

Here, we provide a very basic image processing algorithm based on the Publication "[Post-processing strategies in image scanning microscopy](https://www.sciencedirect.com/science/article/abs/pii/S1046202315001930)" from McGregor et al. 

We cannot guarantee it's correct functionality for correct results in any ISM dataset, since we customized the code to work on our data using low-cost components, such as the MEMS-based laser projector AnyBeam and the industry-grade CMOS camera from Allied Vision. 

You can read more about the setup on the [UC2 ISM page](https://github.com/bionanoimaging/UC2-GIT/tree/master/APPLICATIONS/APP_openISM)


## Workflow 

Figure 3 in the manuscript is based on the file [2020_10_13-HeLa_AF647_ISM_Sony.py](./PYTHON/2020_10_13-HeLa_AF647_ISM_Sony.py)

The general workflow is as follows:

### 1. Load the image dataset: 

```py
#%%-------------------------- FILE I/O ---------------------------------------
ismstack = utils.readismsack(rawfile, darkfile=None, backgroundval=0, myupscale=myscalesize, is_reducetime=is_reducetime, is_denoise=is_denoise, cropsize=cropsize)
#tif.imsave(myfolder + os.path.split(myvideofile)[-1]+'_reducedtime.tif',ismstack)
Nimages = ismstack.shape[0]     # How many images you want to use for the ISM image? (fps~20, illumination~5)
mysize = ismstack.shape[1]    # size from the original 
```

<p align="center">
<img src="./IMAGES/FIG1.png" width="300">
</p>

Along XZ:

<p align="center">
<img src="./IMAGES/FIG2.png" width="300">
</p>

### 2. Create Superconfocal and Brightfield image

```py
#%%-------------------------- Copute Super-Confocal --------------------------
mysuperconfocal = ism.compute_superconfocal(ismstack, is_debug)
mybf = ism.compute_brightfield(ismstack, is_debug)
```

<p align="center">
<img src="./IMAGES/FIG3.png" width="300">
</p>

<p align="center">
<img src="./IMAGES/FIG4.png" width="300">
</p>

### 3. Detect Rotation and Grating Constant

```py
#%%-------------------------- Peak-Detection ---------------------------------
myg, myrot = ism.estimate_ism_pars(testframe=ismstack[2,:,:], mindist=mindist, radius_ft=0.02, is_debug=is_debug)
mindist = myg
rottheta = myrot
my_g_vert = my_g_horz = myg
```

<p align="center">
<img src="./IMAGES/FIG6.png" width="500">
</p>


### 4. Find peaks in each frame and fit a grating to it

```py
    mypeaks_map, mypeaks_pos = ism.get_peaks_from_image(test_frame,mindist=mindist, is_debug=is_debug)
```

<p align="center">
<img src="./IMAGES/FIG5.png" width="300">
</p>

```py
        # There is a lot of magic going on. A set of lines is fit to the points in X and Y
        my_fit_vert, my_vert, my_fit_horz, my_horz, my_g_vert, my_g_horz = ism.fit_illpattern_ism(
                                    mypeaks_pos, polyn = polyn, 
                                    max_std = max_std, searchdist_vert = searchdist_vert, 
                                    searchdist_horz = searchdist_horz, 
                                    is_debug = is_debug, is_fillmissing=is_fillmissing, 
                                    g_vert=my_g_vert_list, g_horz=my_g_horz_list)
        
```

<p align="center">
<img src="./IMAGES/FIG7.png" width="300">
</p>


Optional: Fill missing stripes with most-likely lines according to the period in xy:

```py
        # correct for mismatching grating period along horizontal direction
        # enforce global grating constant to match the fitted one..
        my_fit_horz = ism.adjust_grating_constant(my_fit_horz, 0, is_debug)
        my_fit_vert = ism.adjust_grating_constant(my_fit_vert, 0, is_debug)
```

This is done by fitting a straight line to the gratings center to realign them correctly:
<p align="center">
<img src="./IMAGES/FIG71.png" width="300">
</p>


### 5. Render final point grid

```py
        # generate the deformed illumination grid from the first frame - array => 0-index is y,, 1-index is x
        my_grid, my_grid_index  = ism.generate_illumination_grid(test_frame, my_fit_vert, my_vert, my_fit_horz, my_horz, is_debug)
```       


### 6. Optional: Sort out duplicate peaks

```py
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
```


### 7. Pixel reassignment 

No that we have all the "valid" positions of all illuminating spots, we want to filter the intensity and place the information on a new grid. This is done by iterating over all peak positions, cut out a subroi around their center, mask it and place its integral on the upsampled grid:

```py
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
```

Here a single frame after reassigning the pixel values to a two-fold larger canvas
<p align="center">
<img src="./IMAGES/FIG9_ismsingle.png" width="300">
</p>

Here a sum of several of those frames
<p align="center">
<img src="./IMAGES/FIG9_ismresult.png" width="300">
</p>


### Final image 

Here the sum of all detected peaks, filtered and reassigned: 

<p align="center">
<img src="./IMAGES/23.10.2020_800x600_HeLa_SiR_newSample_3.tif_Subplot.png" width="600">
</p>



## Data 

You can download the dataset from the above example [here](https://drive.google.com/file/d/16X4wF6pqDD2mGebHUPvP6L6UyIBpSL2p/view?usp=sharing)



## Dependencies
```
# install the environment
conda create -n tfgpu2 python=3.6
conda activate tfgpu2

# install spyder and other dependencies
conda install -c anaconda spyder
conda install -c anaconda tifffile
pip install matplotlib
pip install git+https://NanoImagingUser:NanoImagingUser@gitlab.com/bionanoimaging/nanoimagingpack
pip install opencv-python

# start spyder
spyder &
```


## Disclaimer

We cannot guarantee any warranty that this software is fully functional. Use it as your own risk. 
The software was tested on a Mac Book Pro with Catalina, Windows 10 and Ubuntu 16.04 LTS machine equipped with an Nvidia GPU. 

## License

GPLv3, Copyright: Benedict Diederich 2020
## Citation
empty

