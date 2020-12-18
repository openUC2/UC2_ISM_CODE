# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
#from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
import scipy.signal
import cv2
from scipy.ndimage.filters import gaussian_filter

def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def find_shift(im1, im2):
    
    # pixel precision first
    shift, error, diffphase = register_translation(im1, im2, 150)
    #print("Detected pixel offset (y, x): {}".format(shift))
    shift = np.int16(np.floor(shift))
    return shift
    
def find_shift_lattice(lattice, frame):
    
    # pixel precision first
    lattice_psf = gaussian_filter(lattice, sigma=1)
    lattice_psf = lattice_psf * gaussian_filter(1.*(frame>np.mean(frame)),5)
    #plt.imshow(lattice_psf), plt.show()
    shift, error, diffphase = register_translation((lattice_psf/np.max(lattice_psf))**2, (frame/np.max(frame))**2, 150)
    #print("Detected pixel offset (y, x): {}".format(shift))
    shift = (shift)
    #print(shift)
    return shift

def find_shift_sub(im1, im2):
    # subpixel precision
    # pixel precision first
    shift, error, diffphase = register_translation(im1, im2, 150)
    #print("Detected subpixel offset (y, x): {}".format(shift))
    return shift    

def generate_gaussian2D(mysize=10, sigma=1.0, mu=1.0):
    x = np.linspace(- (mysize // 2), mysize // 2, mysize)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    return kernel / kernel.sum()

def mycrosscorr(inputA, inputB):
    return np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(inputA) * np.conj(np.fft.fft2(inputB)))))

def myconv(inputA, inputB):
    return np.fft.ifftshift(np.real(np.fft.ifft2(np.fft.fft2(inputA) * (np.fft.fft2(inputB)))))

def extract2D(input_matrix, newsize = ((300,300)), center=((-1,-1,)), debug=False):
    # This extracts/padds a 2D array with zeros
    # input_matrx - 2D image what we want to crop/pad
    # newsize - new image size (Nx,Ny)
    # center - center coordinates where the ROI needs to get extracted (cx,cy)
    
    
    Nx_old, Ny_old = input_matrix.shape
    
    # This is the new size of the outputmatrix
    Nx_new, Ny_new = newsize
    
    # Center-coordinates of new image
    cx_new = np.floor(Nx_new/2)
    cy_new = np.floor(Ny_new/2)
    
    # where should we crop out the image?
    # Either provide your own coodinates or keep it as the center of the image     
    if center[0] == -1:
        # define center of new image in x-direction
        cx_old = np.floor(Nx_old/2)
    else:
        cx_old = center[0]
        
    # where should we crop out the image?     
    if center[1] == -1:
        # define center of new image in x-direction
        cy_old = np.floor(Ny_old/2)
    else:
        cy_old = center[1]
        
    # We have at least two cases: 
    # 1: Image is smaller than crop region
    # 2: Image is bigger then crop region
    # 2a: We can simply crop out the image
    # 2b: Center region is not within the image in x-direction
    # 2c: Center region is not within the image in y-direction 
    
    # eventually we need to add one pixel if the crop-size is odd
    odd_x = np.mod(Nx_new,2)
    odd_y = np.mod(Ny_new,2)    
    
    if(np.sum(np.int16(newsize>input_matrix.shape))>0):
        if(debug): print('Simply padding the array')
        if(debug): print('Not correclty implemented!')
        # Case1 - padd the image with zeros
        myimage_new = np.zeros((Nx_new, Ny_new)) # create new image
    
        # we place the old image inside the new one
        myimage_new[int(cx_new-cx_old):int(cx_new+cx_old),int(cy_new-cy_old):int(cy_new+cy_old)]=input_matrix
    else:
        if(debug): print('Need to crop!')
    
        
        # Calculate crpping coordinates
        x_min = np.int16(cx_old-np.floor(Nx_new/2))
        x_max = np.int16(cx_old+np.floor(Nx_new/2)+odd_x)
        y_min = np.int16(cy_old-np.floor(Ny_new/2))
        y_max = np.int16(cy_old+np.floor(Ny_new/2)+odd_y)
        
        # Case 2 - We have to cop the image 
        myimage_new = np.zeros((Nx_new, Ny_new)) # create new image
        
        # Case 2a - We can simply crop the image
        if(Nx_old<=Nx_new and Nx_old<=Nx_new):
            if(debug): print('Case 2a, easy!')
            myimage_new = input_matrix[int(cx_old-np.floor(Nx_new)):int(cx_old+np.floor(Nx_new)+odd_x),int(cy_old-np.floor(Ny_new)):int(cy_old+np.floor(Ny_new)+odd_y)]
        else:
            # Case 2b - We are at the edge in +/- x-direction
            # We are at the left edge x<0
            posx=posy=0
            if(x_min<0):
                posx = x_min # keep the pixels of how much we shifted it
                x_min = 0
            # We are at the right edge x>x_end
            if(x_max>Nx_old):
                posx = x_max # keep the pixels of how much we shifted it
                x_min = Nx_old
               
            # crop in X-direction            
            input_matrix = input_matrix[x_min:x_max,:]
    
    
            # Case 2c - We are at the edge in +/- x-direction
            # We are at the upper edge y<0
            if(y_min<0):
                posy = y_min # keep the pixels of how much we shifted it
                y_min = 0
            # We are at the lower edge y<0
            if(y_max>Ny_old):
                posy = y_max # keep the pixels of how much we shifted it
                y_min = Ny_old
      
            # crop in Y-direction            
            input_matrix = input_matrix[:,y_min:y_max]
            
            # Eventually shift the input matrix bback to the original coordinates
            newshape = input_matrix.shape; newx,newy = newshape[0], newshape[1]
            myimage_new = np.zeros((Nx_new, Ny_new))
            
            # place the input_matrix in side the image 
            # we need to differentiate 4 cases
            myimage_new[0:newx, 0:newy] = input_matrix
    
            # shift the image back, so that the center of the old image still is in the cneter of the new one
            myimage_new = np.roll(myimage_new, -posx, axis=0)
            myimage_new = np.roll(myimage_new, -posy, axis=1)
    
        if(debug): plt.imshow(myimage_new)
        if(debug): print(myimage_new.shape)
        return myimage_new

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    #vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    return img

def warp_flow(img, flow):
    flow_adjust = 1
    h, w = flow.shape[:2]
    flow = -flow * flow_adjust
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def cross_image(im1, im2):
   # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = im1.astype('float')
   im2_gray = im2.astype('float')

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')


def alignImages(im1, im2):  
    MAX_FEATURES = 50
    GOOD_MATCH_PERCENT = 0.15
     
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
       
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
       
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
     
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
     
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
     
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
           
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
     
    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
       
    return im1Reg, h
