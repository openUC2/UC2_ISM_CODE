# -*- coding: utf-8 -*-

import skvideo.io
import skvideo.datasets
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt 

myvideofile = '/Users/bene/Dropbox/Confocal/with cubes/vid1.h264'
myvideofile = '/Users/bene/Dropbox/Confocal/Probe mit Huawei P9/Haselpollen-groß-1-Huawei.mp4'
myvideopath = '/Users/bene/Dropbox/Confocal/15.11-HUAWEI/'
myvideofile = '2018-11-15 14.33.08.mp4'
outputfile = myvideofile+'.tif'
myvideofile = myvideopath+myvideofile
videogen = skvideo.io.vreader(myvideofile)

for frame in videogen:
    gray = np.mean(frame, 2)
    tif.imsave(outputfile, np.uint8(gray), append=True, bigtiff=True) #compression='lzw',     

    
    #tif.imsave(outputfile, np.uint8(gray), append=True, bigtiff=True) #compression='lzw', 
