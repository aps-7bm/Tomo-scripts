#!/usr/bin/env python
"""
Code developed in Cycle 2018-3 to calculate the center of rotation automatically both a 180 and 350 degree spin. 
"""

import os
import sys
import cv2
import copy
import time
import h5py
import glob
import psutil
import tomopy
import matplotlib 
import numpy as np
from PIL import Image
import multiprocessing
import dxchange.writer
matplotlib.use('tkAgg')
from scipy import ndimage
import matplotlib.pyplot as plt
import tomopy.util.dtype as dtype
from tomopy.util.misc import fft2
from joblib import Parallel, delayed
from tomopy.misc.morph import downsample
#from skimage.filters import unsharp_mask
import tomoPyScriptUtilities.BeamHardening_Transmission as bh

# USER SETTINGS ######################################################################################

#file information
main_folder = '/mnt/SprayData/Cycle_2018_3/Tomography/Fister/t2_drycell_5x/'
data_key = 't2_pam_bottom_wet_latertime' #'ALL' if all folders within main_folder should be processed, otherwise specify part of dataset name
cfile = '/mnt/SprayData/Cycle_2018_3/Tomography/Fister/Cycle2018_3_Centers_Fister.txt' #where to save calculated centers of rotation
magnification = 5 #specify magnification if it isn't in the HDF5 file

#Recon settings
pad_images = 'edge' #None, zero, edge
pad_fraction_width = 0.5 # fraction of image width to pad on each side, if pad_images is not None
remove_pad = 1.0 # remove fraction of padded region to make output files smaller?
algorithm = 'gridrec' # check tomopy docs for options
recon_filter = 'parzen' # check tomopy docs for options. this is for gridrec

#crop image columns - None if no cropping required
start_col = None
end_col = None

#automatic center of rotation at search row
find_center_automatically = True #using Nghia Vo's method
center_search_row = 400 #row at which to perform the center of rotation diagnosis
center_location = 'left' #right or left of image midplane, if spinning 360
hist_thresh = 0.01 #used to increase contrast by streching histogram
contrast_thresh = 0.03 #contrast threshold to set on sinogram row that is being matched 

#Paramters for Nghia Vo's automatic center of rotation method
smin = -500 #Minimum coarse search radius. Reference to the horizontal center of the sinogram.
smax = 0 #Maximum coarse search radius. Reference to the horizontal center of the sinogram.
sstep = 1   #Step size for coarse search radius
ratio = .7  #The ratio between the FOV of the camera and the size of object. It’s used to generate the mask.
drop = 20   #Drop lines around vertical center of the mask.
outputsino = True #output sinogram images for troubleshooting
smooth = True #Whether to apply additional smoothing or not.
srad = 10 #Fine search radius.
step = 0.2 #Step of fine searching.       
        
#write images around desired center of rotation
write_cor_images = True
center_guess = 1876 #if None, get center from automatated calculation
search_margin = 30 # +/- number of pixels to search from search_margin
search_step = 0.5 #step size of search

#I0 region (for normalizing when doing conversion to path length)
I0_normalize = True
I0_box_size = 40 # pixels 
I0_box_loc = 'center right' # 'lower/upper/center left/right/middle'. Skips first two rows due to bad pixel values.

# Phase retreival parameters
phase_retrieval = False
margin = 0  # Num slices at ends to not write (due to phase retrieval boundary effects)
z = 5 # propagation distance (cm)
eng = 62 # Flux-weighted mean energy of white beam (keV)
alpha = 1e-3 #6e-4 # Smaller number is smoother, larger number is less phase retrieval correction

# Max number of CPUs for computation and I/O operations
ncore=multiprocessing.cpu_count()-1

# END USER SETTINGS ###################################################################################

'''FUNCTION DEFINITIONS BELOW'''

#Read in HDF5 projection data from tomography scan  
def fread_projection_images(data_filename,row_slice= slice(None,None,1),column_slice=slice(None,None,1)):
    '''Loads the flyscan data from HDF5 files.
    '''    
    t0 = time.time()   
    with h5py.File(data_filename,'r') as data_hdf:
        data = data_hdf['Prefiltered_images'][:,row_slice,column_slice]
    print(('\tElapsed reading data: %i sec\n' % (time.time()-t0)))
    return data.astype(np.float32)

#Read in bright and dark fields 
def fread_bright_dark_images(data_filename,bright_filename,dark_filename,row_slice= slice(None,None,1),column_slice=slice(None,None,1),bright_ratio=1):
    '''Loads the bright and dark images.
    '''    
    t0 = time.time()   
    if bright_filename and 'tif' in bright_filename:
        bright_data_im = Image.open(bright_filename)
        bright_data_pre = np.array(bright_data_im) 
    elif bright_filename and 'hdf5' in bright_filename: 
        with h5py.File(bright_filename,'r') as bright_hdf:
            bright_data_pre = bright_hdf['Prefiltered_images'][...]
    else:
        with h5py.File(data_filename,'r') as data_hdf:
            bright_data_pre = data_hdf['Bright_field'][...]
    bright_data = bright_data_pre[row_slice,column_slice]
    #remove high intensity pixels in the top-left corner from bright field
    #bright_data = tomopy.misc.corr.remove_outlier1d(bright_data,dif=1000)  

    if dark_filename and 'tif' in dark_filename:
        dark_data_im = Image.open(dark_filename)
        dark_data_pre = np.array(dark_data_im)
    elif dark_filename and 'hdf5' in dark_filename: 
        with h5py.File(dark_filename,'r') as dark_hdf:
            dark_data_pre = dark_hdf['Prefiltered_images'][...]
    else:
        with h5py.File(data_filename,'r') as data_hdf:
            dark_data_pre = data_hdf['Dark_field'][...]
    dark_data = dark_data_pre[row_slice,column_slice]
    #remove high intensity pixels from dark image
    #dark_data = tomopy.misc.corr.remove_outlier1d(dark_data,dif=1000)
    print(('\tElapsed reading bright/dark: %i sec\n' % (time.time()-t0)))
    return (bright_data.astype(np.float32) * bright_ratio, dark_data.astype(np.float32))

#Find IO region
def fI0_slices():
    '''Gives slice objects for the rows and columns for the I0 region.
    '''
    #Find the shape of the underlying data
    dataset_shape = fget_dataset_shape(data_filename)
    M = dataset_shape[-2]
    N = dataset_shape[-1]
    mid_row = int((M+I0_box_size)/2.)
    mid_col = int((N+I0_box_size)/2.)
    #Make the location string lower case and split it
    I0_box_split=I0_box_loc.lower().split()
    #Figure out the vertical location
    slice_M = None
    slice_N = None

    if I0_box_split[0] == 'lower':
        slice_M = slice(-I0_box_size,None,1)
    #avoid very top row due to bad pixel values in top left corner
    elif I0_box_split[0] == 'upper':
        slice_M = slice(2,I0_box_size,1)
    elif I0_box_split[0] == 'center':
        slice_M = slice(mid_row-I0_box_size,mid_row,1)
    else:
        raise ValueError()
    #Figure out the horizontal location
    if I0_box_split[1] == 'right':
        slice_N = slice(-I0_box_size,None,1)
    elif I0_box_split[1] == 'left':
        slice_N = slice(0,I0_box_size,1)
    elif I0_box_split[1] == 'middle':
        slice_N = slice(mid_col-I0_box_size,mid_col,1)
    else:
        raise ValueError()
    return (slice_M,slice_N)

#Calculate I0 region for each projection
def fI0_calculation(row_slices,column_slices,bright_ratio=1):
    '''Computes the I0 region.
    '''
    #Read in the projection data
    proj_data = fread_projection_images(data_filename,row_slices,column_slices)
    #Read in the bright and dark image subsets
    bright_data,dark_data = fread_bright_dark_images(data_filename,bright_filename,dark_filename,row_slices,column_slices, bright_ratio)
    #Perform bright and dark corrections
    proj_data = tomopy.normalize(proj_data,bright_data,dark_data)
    #Average: use 64 bit for accumulator, then down-convert to 32 bit
    return_vals = np.mean(proj_data,axis=(-2,-1),dtype=np.float64).astype(np.float32)
    #Norm by the maximum value
    return return_vals / np.max(return_vals)

#normalize by I0 region
def fnormalize_image_brightness(projection_data,I0_values):
    '''Uses a small region outside the sample to normalize for shifts
    in image intensity.
    '''
    t0=time.time()
    print("Normalizing by I0 box ...")
    projection_data /= I0_values[:,None,None]
    print(("\tElapsed normalizing image by I0: {0:6.3f} sec".format(time.time() - t0)))
    return projection_data

#Get shape of tomography data
def fget_dataset_shape(data_filename,row_slice= slice(None,None,1),column_slice=slice(None,None,1)):
    with h5py.File(data_filename,'r') as data_hdf:
        return data_hdf['Prefiltered_images'].shape

#remove stripes from projection data 
def fremove_stripes(dataset,method='sf'):
    '''Method for removing stripes from the dataset.
    '''
    t0=time.time()
    print("Removing stripes/rings...")
    if method == 'fw':
        dataset = tomopy.prep.stripe.remove_stripe_fw(dataset,level=8,wname='sym16',sigma=1,pad=True,ncore=ncore) # Fourier Wavelet method
    elif method == 'ti':
        dataset = tomopy.prep.stripe.remove_stripe_ti(dataset) # Titarenko algorithm
    else:
        dataset = tomopy.prep.stripe.remove_stripe_sf(dataset, 10, ncore=ncore) # Smoothing filter stripe removal
    print(("\tElapsed removing chunk stripes: {0:6.3f} sec".format(time.time() - t0)))
    return dataset

#phase retrieval algorithm, if using
def fapply_phase_retrieval(dataset):
    t0=time.time()
    print("Apply phase retrieval...")
    dataset = tomopy.prep.phase.retrieve_phase(dataset,pixel_size=pixel_size,dist=z,energy=eng,alpha=alpha,pad=True,ncore=ncore)
    print(("\tElapsed in phase retrieval: {0:6.3f} sec".format(time.time() - t0)))
    return dataset

#re-register data for 360 deg scans onto 0-180 deg
def freregister_data(dataset,theta,rotation_center,center_location,delta_theta):
    #If we have > 180 degrees of projections, reregister
    if np.max(theta) - np.min(theta) > np.pi:
        print("\nRe-registering projections for span > 180 degrees...")    
        #Find starting point for angles > 180 degrees, account for roundoff error
        index_180 = np.argmax(theta >= (np.pi - delta_theta / 10.0)) #Returns first argument where this is True
        #Make new theta array
        new_theta = theta[:index_180]
        #Verify that the angles before and after 180 degrees match closely
        if (np.allclose(new_theta+np.pi,theta[index_180:2*index_180],delta_theta / 10.0)):
            print("Angles before and after 180 degrees match.")
        else:          
            raise ValueError()
        #Round the center of rotation to the nearest half pixel.
        rotation_center = np.round(rotation_center * 2.0) / 2.0
        #Find new width of array
        old_image_center = (dataset.shape[2] - 1.0) / 2.0      #In between middle two pixels
        overlap_width = float(dataset.shape[2])  - 2.0* np.abs(old_image_center - rotation_center)
        new_width = dataset.shape[2]  + int(2.0 * np.abs(old_image_center - rotation_center))
        new_center = float(new_width - 1) / 2.0   
        #Make new dataset array 
        data_rereg = tomopy.sino_360_t0_180(dataset,overlap_width,center_location)
        data_size = float((dataset.nbytes)/1e9)
        del(dataset)
        rereg_size = float((data_rereg.nbytes)/1e9)
        return(data_rereg,new_theta,new_center)
    else:
        print("No reregistration of angles needed.")
        return(dataset.copy(),theta.copy(),rotation_center.copy())

#pad images before reconstruction
def fpad_image(dataset,center_value,pad_fraction_width):
    '''Pads images with the desired values.
    '''
    if pad_images is None:
        return (dataset,center_value)
    
    t0=time.time()
    prev_width = dataset.shape[2]    
    print("\nPadding array...", dataset.shape)
    pad_pixels=int(np.ceil(prev_width*pad_fraction_width))
    if pad_images == 'constant':
        padded_dataset = tomopy.pad(dataset,npad = pad_pixels, axis = 2, mode='constant',ncore=ncore,constant_values=1.0)
    elif pad_images == 'edge':
        padded_dataset = tomopy.pad(dataset,npad = pad_pixels,axis = 2, mode='edge',ncore=ncore)
    else:
        raise ValueError("pad_images = "+pad_images+" not understood!")    
    print("\tNew size:",padded_dataset.shape)
    print('\tElapsed %i sec\n' % (time.time()-t0))
    #Center offset
    center_offset = (padded_dataset.shape[2]-dataset.shape[2])/2
    pad_size = float((padded_dataset.nbytes)/1e9)
    return (padded_dataset,center_value + center_offset)

#write out the tiffstack of reconstructed images
def fwrite_tiff_stack(recon_dataset,smin,total_rows):
    #Write 32-bit tiff stack out
    t0=time.time()
    #Leave out margin rows in case we did phase retrieval
    start_write_row = 1 if (smin and margin) else 0
    last_write_row = -1 if (recon_dataset.shape[0] - total_rows and margin) else 0
    start_write_row += start_row + smin
    last_write_row += start_row + smin + recon_dataset.shape[0]
    print(("Writing slices {0:d}-{1:d} to {2:s}...".format(
             start_write_row, last_write_row - 1,output_path_tiffstack)))
    for i in range(start_write_row,last_write_row):
        j = i - start_row - smin
        dxchange.write_tiff(recon_dataset[j,...],fname=output_path_tiffstack+'img_{0:04d}'.format(i),overwrite=True)              
    print('\tElapsed %i sec\n' % (time.time()-t0))  

#reconstruct projections
def freconstruct_projections(dataset,theta,center,final_size):
    #Tomographic reconstruction
    t0=time.time()
    print("\nReconstructing...")
    data_recon = tomopy.recon(dataset, theta,center=center,algorithm=algorithm,ncore=ncore,filter_name=recon_filter)
    print('\tElapsed %i sec\n' % (time.time()-t0))
    del(dataset,theta,center)      
    #Crop the image to the desired size
    #print(("Cropping volume from size {0:d} to size {1:d}".format(data_recon.shape[1],final_size)))
    crop_margin = int((data_recon.shape[1] - final_size) / 2.0)
    if crop_margin > 0:
        data_recon = data_recon[:,crop_margin:crop_margin+final_size,crop_margin:crop_margin+final_size]
    #Rearrange output (Dan notices that the images are flipped vertically!)
    data_recon = np.fliplr(data_recon) 
    #print(("Chunk output stats: ", data_recon.shape))
    #print(("Nans: ",np.sum(np.isnan(data_recon)),'/',np.product(data_recon.shape)))
    #print(("Infs: ",np.sum(np.isinf(data_recon)),'/',np.product(data_recon.shape)))     
    #Mask NaNs and INFs from the output data.
    data_recon[np.isnan(data_recon)]=0
    data_recon[np.isinf(data_recon)]=np.nanmax(data_recon)*np.sign(data_recon[np.isinf(data_recon)])
    print(('\tTotal elapsed to reconstruct subset = {0:3.3f} s.'.format(time.time() - t0)))
    return data_recon

#remove outliers with a median filter      
def remove_outliers(data,dif,size):
    out = tomopy.misc.corr.remove_outlier1d(data,dif=dif,size=size,axis=1)
    return out

#write out images with different center of rotation guesses
def write_center_images(data,theta,guess,cor,step_theta,path):
    old_image_center = (data.shape[2] - 1.0) / 2.0 #In between middle two pixels
    overlap_width = float(data.shape[2])  - 2.0* np.abs(old_image_center - cor)
    desired_size = data.shape[2]  + int(2.0 * np.abs(old_image_center - cor))
    reg_data, reg_theta, reg_center = freregister_data(data,theta,guess,center_location,step_theta)
    del(data)
    #Mask huge values, NaNs and INFs from the filtered data
    maxv=1e5
    reg_data[np.isnan(reg_data)]=0
    reg_data[np.isinf(reg_data)]=np.nanmax(reg_data)*np.sign(reg_data[np.isinf(reg_data)])
    reg_data[np.abs(reg_data)>maxv]=maxv*np.sign(reg_data[np.abs(reg_data)>maxv])
    (pad_data, pad_center) = fpad_image(reg_data,reg_center,pad_fraction_width=pad_fraction_width)
    del(reg_data)
    data_recon = freconstruct_projections(pad_data,theta,pad_center,desired_size)
    del(pad_data)
    data_recon = tomopy.remove_ring(data_recon.copy(),ncore=ncore)
    dxchange.write_tiff(data_recon[0,...],fname = path+'%0.1f'%guess,overwrite=True)
    return 

# Nghia Vo's code from TomoPy v.1.3.0 copied below with minor edits that allow for COR diagnosing ###############################

def find_center_vo(tomo,in180,ind=None, smin=-50, smax=50, sstep = 5, outputsino = True, srad=6, step=0.5, ratio=0.5, drop=20, smooth=False):
    """
    Find rotation axis location using Nghia Vo's method. :cite:`Vo:14`.

    Parameters
    ----------
    tomo : ndarray
        3D tomographic data.
    ind : int, optional
        Index of the slice to be used for reconstruction.
    smin, smax : int, optional
        Coarse search radius. Reference to the horizontal center of the sinogram.
    sstep: int, optional
        Step size used in coarse search 
    srad : float, optional
        Fine search radius.
    step : float, optional
        Step of fine searching.
    outputsino: bool, optional
        Output sinograms into file path for diagnosing center of rotation
    ratio : float, optional
        The ratio between the FOV of the camera and the size of object.
        It's used to generate the mask.
    drop : int, optional
        Drop lines around vertical center of the mask.
    smooth : bool, optional
        Whether to apply additional smoothing or not.

    Returns
    -------
    float
        Rotation axis location.
    """
    tomo = dtype.as_float32(tomo)

    if ind is None:
        ind = tomo.shape[1] // 2
    _tomo_pre = tomo[:, ind, :]
    _tomo = copy.deepcopy(_tomo_pre)
        
    #improve contrast by tightening up histogram 
    hist = np.histogram(_tomo_pre,bins=100)
    histg = abs(np.gradient(hist[0]))
    maxval = np.max(histg)
    l = len(histg)
    #set thresholds based on gradient peaks 
    threshl = hist[1][np.argmax(histg>hist_thresh*maxval)]
    lh = len(histg[int(l/2):])
    histgr = np.flip(histg)
    threshh = hist[1][int(l-(np.argmax(histgr>hist_thresh*maxval)))]
    plt.figure(1,figsize=(11,8.5))
    plt.title('Original Histogram')
    plt.plot(hist[1][1:],hist[0],'k-',lw=2)
    plt.xlabel('Intensity')
    plt.ylabel('Counts')
    plt.axvline(threshl,ls='--',color='b',label = 'Lower threshold')
    plt.axvline(threshh,ls='--',color = 'g', label = 'Upper threshold')
    plt.ylim(0,1.1*np.max(hist[0]))
    plt.legend(loc='best')
    #plt.show()
    #exit()
    plt.savefig(file_path+'SinoHist.png')
    plt.close(1)
    
    _tomo[_tomo_pre<threshl]=  threshl
    _tomo[_tomo_pre>threshh] = threshh
     
    # Reduce noise by smooth filters. Use different filters for coarse and fine search
    if smooth:
        _tomo_cs_pre = ndimage.filters.gaussian_filter(_tomo,(3,3))
        _tomo_fs_pre = ndimage.filters.gaussian_filter(_tomo,(2,2))
        #use unsharp mask to accentuate features
        lap_cs = cv2.Laplacian(_tomo_cs_pre,cv2.CV_32F)
        lap_fs = cv2.Laplacian(_tomo_fs_pre,cv2.CV_32F)
        strength = 2
        _tomo_cs = _tomo_cs_pre - strength*lap_cs
        _tomo_fs = _tomo_fs_pre - strength*lap_fs
    else:
        _tomo_cs = _tomo
        _tomo_fs = _tomo        
    #_tomo_cs = ndimage.filters.gaussian_filter(_tomo, (3, 1)) if smooth else _tomo
    #_tomo_fs = ndimage.filters.median_filter(_tomo, (2, 2)) if smooth else _tomo

    # Coarse and fine searches for finding the rotation center.
    if _tomo.shape[0] * _tomo.shape[1] > 4e6:  # If data is large (>2kx2k)
        _tomo_coarse = downsample(np.expand_dims(_tomo_cs,1), level=2)[:, 0, :]
        sc=4
        init_cen = _search_coarse(_tomo_coarse,in180,sc,smin / 4.0, smax / 4.0, sstep, ratio, drop,outputsino)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen*4, ratio, drop)
    else:
        sc=1
        init_cen = _search_coarse(_tomo_cs,in180,sc, smin, smax,sstep, ratio, drop,outputsino)
        fine_cen = _search_fine(_tomo_fs, srad, step, init_cen, ratio, drop)
    return fine_cen

def _search_coarse(sino,in180,sc,smin, smax, sstep, ratio, drop,outputsino):
    """
    Coarse search for finding the rotation center.
    """
    (Nrow, Ncol) = sino.shape
    centerfliplr = (Ncol - 1.0) / 2.0

    # Copy the sinogram and flip left right, the purpose is to
    # make a full [0;2Pi] sinogram
    _copy_sino = np.fliplr(sino[1:])

    # This image is used for compensating the shift of sinogram 2
    temp_img = np.zeros((Nrow - 1, Ncol), dtype='float32')
    temp_img[:] = np.flipud(sino)[1:]
    
    #plot sinograms
    '''
    plt.figure(1)
    plt.imshow(sino,cmap='gray')
    plt.figure(2)
    plt.imshow(_copy_sino,cmap='gray')
    plt.figure(3)
    plt.imshow(temp_img,cmap='gray')
    plt.show()
    '''

    # Start coarse search 
    listshift = np.arange(smin, smax + 1,sstep).astype(int)
    listmetric = np.zeros(len(listshift), dtype='float32')
    mask = _create_mask(2 * Nrow - 1, Ncol, 0.5 * ratio * Ncol, drop)
    sino_sino = np.vstack((sino, _copy_sino))
    abs_fft2_sino = np.empty_like(sino_sino)
    for (num,i) in enumerate(listshift):
        _sino = sino_sino[len(sino):]
        _sino[...] = np.roll(_copy_sino, i, axis=1)
        if i >= 0:
            _sino[:, 0:i] = temp_img[:, 0:i]
        else:
            _sino[:, i:] = temp_img[:, i:]
        fft2sino = np.fft.fftshift(fft2(sino_sino))
        np.abs(fft2sino, out=abs_fft2_sino)
        abs_fft2_sino *= mask
        listmetric[num] = abs_fft2_sino.sum()
        if outputsino ==True:
            dxchange.write_tiff(sino_sino,fname=file_path+'Sinograms_coarse/img_%i'%num,overwrite=True) 
            #dxchange.write_tiff(abs_fft2_sino,fname=file_path+'FFT/img_%i'%num,overwrite=True)                           
    minpos = np.argmin(listmetric)
    normetric = (listmetric-listmetric.min())/(listmetric.max()-listmetric.min())
    out = start_col + sc*(centerfliplr + listshift/2)
    cent = start_col + sc*(centerfliplr + listshift[minpos]/2)
    fig = plt.figure(1,figsize=(8.5,11))
    plt.plot(out,normetric,'ok--',label='center=%0.2f'%cent)
    plt.xlabel('center loc. [px]')
    plt.ylabel('metric [a.u.]')
    plt.legend(loc='best')
    fig.savefig(file_path+'VO_coarse.png')   
    plt.close(fig)     
    return centerfliplr + listshift[minpos] / 2.0

def _search_fine(sino, srad, step, init_cen, ratio, drop):
    """
    Fine search for finding the rotation center.
    """
    Nrow, Ncol = sino.shape
    centerfliplr = (Ncol + 1.0) / 2.0 - 1.0
    # Use to shift the sinogram 2 to the raw CoR.
    shiftsino = np.int16(2 * (init_cen - centerfliplr))
    _copy_sino = np.roll(np.fliplr(sino[1:]), shiftsino, axis=1)
    if init_cen <= centerfliplr:
        lefttake = np.int16(np.ceil(srad + 1))
        righttake = np.int16(np.floor(2 * init_cen - srad - 1))
    else:
        lefttake = np.int16(np.ceil(init_cen - (Ncol - 1 - init_cen) + srad + 1))
        righttake = np.int16(np.floor(Ncol - 1 - srad - 1))
    Ncol1 = righttake - lefttake + 1
    mask = _create_mask(2 * Nrow - 1, Ncol1, 0.5 * ratio * Ncol, drop)
    numshift = np.int16((2 * srad) / step) + 1
    listshift = np.linspace(-srad, srad, num=numshift)
    listmetric = np.zeros(len(listshift), dtype='float32')
    factor1 = np.mean(sino[-1, lefttake:righttake])
    factor2 = np.mean(_copy_sino[0, lefttake:righttake])
    _copy_sino = _copy_sino * factor1 / factor2
    num1 = 0
    for (num,i) in enumerate(listshift):
        _sino = ndimage.interpolation.shift(
            _copy_sino, (0, i),prefilter=False)
        sinojoin = np.vstack((sino, _sino))
        listmetric[num] = np.sum(np.abs(np.fft.fftshift(fft2(sinojoin[:, lefttake:righttake + 1]))) * mask)
        if outputsino ==True:
            dxchange.write_tiff(sinojoin[:, lefttake:righttake + 1],fname=file_path+'Sinograms_fine/img_%i'%num,overwrite=True)
    minpos = np.argmin(listmetric)
    normetric = (listmetric-listmetric.min())/(listmetric.max()-listmetric.min())
    out = start_col + init_cen + listshift/2
    cent = start_col + init_cen + listshift[minpos]/2    
    fig = plt.figure(1,figsize=(8.5,11))
    plt.plot(out,normetric,'og--',label='center=%0.2f'%cent)
    plt.xlabel('center loc. [px]')
    plt.ylabel('metric [a.u.]')
    plt.legend(loc='best')
    fig.savefig(file_path+'VO_fine.png')   
    plt.close(fig)  
    return init_cen + listshift[minpos] / 2.0

def _create_mask(nrow, ncol, radius, drop):
    du = 1.0 / ncol
    dv = (nrow - 1.0) / (nrow * 2.0 * np.pi)
    centerrow = np.int16(np.ceil(nrow / 2) - 1)
    centercol = np.int16(np.ceil(ncol / 2) - 1)
    mask = np.zeros((nrow, ncol), dtype='float32')
    for i in range(nrow):
        num1 = np.round(((i - centerrow) * dv / radius) / du)
        (p1, p2) = np.int16(np.clip(np.sort(
            (-int(num1) + centercol, num1 + centercol)), 0, ncol - 1))
        mask[i, p1:p2 + 1] = np.ones(p2 - p1 + 1, dtype='float32')
    if drop < centerrow:
        mask[centerrow - drop:centerrow + drop + 1,
             :] = np.zeros((2 * drop + 1, ncol), dtype='float32')
    mask[:, centercol-1:centercol+2] = np.zeros((nrow, 3), dtype='float32')
    return mask

#calculate correct amount of images to load at once for system    
def calculate_chunksize_cor(data,ims):
    #automated way to calculate the chunk size 
    mem = psutil.virtual_memory()
    avmem = mem.available/(1e9) #system memory that is available and not swap, in GB
    #assume that we want 1/4 of the system memory to be free for other things and can therefore use 3/4 of it
    threshold = avmem*(3/4.)
    #calculate about how much RAM is going to be needed for the whole recon 
    set_size = (4*np.size(data))/(1e9) #dataset gets converted to 32 bit i.e. 4 bytes per bit
    pad_size = (set_size*pad_fraction_width)*2+set_size #size after padding occurs
    recon_size = pad_size - (set_size*(pad_fraction_width*remove_pad))*2 #size of reconstruction
    #memory required if nothing gets deleted in time on any of the usable cores
    memreq = (set_size*2+pad_size+recon_size)*ncore*ims
    chunkfrac = threshold/memreq
    #find chunksize that can be supported in system memory using n cores
    rchunk = int(chunkfrac*ims)
    print(rchunk,ims)
    return rchunk
        
#########################################################################################################################
# Main function to do tomoPy reconstruction. Input parameters are defined at the top of the script.
# This version will load ALL the images and work on chunks of specified size in RAM.
# Chunks are overlapped by some 'margin' to avoid edge phase retrieval errors.
### MAIN ################################################################################################################

def ftomopy_script():

    print(("TomoPy version = " + tomopy.__version__))
    
    #read in necessary properties from HDF5 file
    with h5py.File(data_filename,'r') as data_hdf:
        bright_exp = data_hdf['measurement']['instrument']['detector']['brightfield_exposure_time'][...] #bright field exposure time, units of seconds 
        exp = data_hdf['measurement']['instrument']['detector']['exposure_time'][...] #images exposure time, units of seconds
        bright_ratio = exp/bright_exp #multiply bright field intensity values by this ratio 
        #load in magnification
        try:
            mag = data_hdf['measurement']['instrument']['detector']['magnification'][...]
        except: 
            try:
                mag = data_hdf['measurement']['instrument']['detector']['setup']['Magnification'][...]            
            except KeyError:
                mag = magnification
        cam_model = str(data_hdf['measurement']['instrument']['detector']['model'][...])
        #retrieve pixel sizes from camera model name
        if cam_model == 'Grasshopper3 GS3-U3-123S6M':
            pixel_size = 3.45e-4/mag
        elif cam_model == 'Grasshopper3 GS3-U3-23S6M':
            pixel_size = 5.86e-4/mag
        #figure out theta range        
        start_theta = np.deg2rad(data_hdf['measurement']['instrument']['sample_position']['Sample_Theta_Start'][...])
        end_theta = np.deg2rad(data_hdf['measurement']['instrument']['sample_position']['Sample_Theta_End'][...])
        step_theta = np.deg2rad(data_hdf['measurement']['instrument']['sample_position']['Sample_Theta_Step'][...])
        #Span of angles for reconstruction; units radians
        theta = np.arange(start_theta,end_theta+step_theta,step_theta)       
       
    #set slices to reconstruct 
    row_slice = center_search_row
    column_slice = slice(None,None,1)
    
    #Import one projection image
    projection_data_full = fread_projection_images(data_filename,row_slice,column_slice)
    #expand dimensions so that data has 3 axes
    projection_data_full = np.expand_dims(projection_data_full,axis=1)

    #Load the bright and dark images.  Do over the same subset of the image as the original data.
    bright_median, dark_median = fread_bright_dark_images(data_filename,bright_filename,dark_filename,row_slice,column_slice,bright_ratio)

    #Normalize the projection images by the bright and dark fields
    projection_data_full = tomopy.normalize(projection_data_full,bright_median,dark_median)

    #Find the normalization values
    if I0_normalize:     
        I0_row_slice,I0_column_slice = fI0_slices()
        I0_values = fI0_calculation(I0_row_slice, I0_column_slice,bright_ratio)  
        #Normalize by the I0 region. 
        projection_data_full = fnormalize_image_brightness(projection_data_full,I0_values)
   
#    plt.figure(1)
#    plt.imshow(projection_data_full[:,0,:],cmap='gray') #plot sinogram
#    plt.title('After Normalization')
#    plt.colorbar()
#    plt.show()       

    #Remove stripes
    projection_data_full = fremove_stripes(projection_data_full)
    #Take log
    projection_data_full = tomopy.minus_log(projection_data_full)
    #Apply phase retrieval, if desired
    if phase_retrieval:
        projection_data_full = fapply_phase_retrieval(projection_data_full)
        
    '''          
    plt.figure(1)
    plt.imshow(sub_data[:,0,:],cmap='gray',vmin=0,vmax=1) #plot sinogram
    plt.title('Slice for COR Calculation')
    plt.colorbar()
    plt.show()
    '''
    
    if find_center_automatically:
        '''
        Find center automatically using TomoPy Nghia Vo's method
        '''
        print("Diagnosing the center of rotation automatically...")
        in180 = np.argmax(theta >= (np.pi - step_theta / 10.0)) #Returns first argument where this is True
        shift = 0
        sino_roll = np.roll(projection_data_full[...,start_col:end_col],shift,axis=0)
        sino = sino_roll[:in180+1,...]
        meani = np.abs(np.mean(sino[-5:,0,...],axis=0))
        diff = meani.max()-meani.min()
        #look at the gradient at the boundary

        while diff<contrast_thresh:
            shift+=100
            sino_roll = np.roll(projection_data_full[...,start_col:end_col],shift,axis=0)
            sino = sino_roll[:in180+1,...]
            meani = np.abs(np.mean(sino[-5:,0,...],axis=0))
            diff = meani.max()-meani.min()
        print("final shift is %i"%shift)           
        cor = find_center_vo(sino,in180,ind=0, smin=smin, smax=smax, sstep = sstep, outputsino = outputsino, srad=srad, step=step,
                   ratio=ratio, drop=drop, smooth=smooth) + start_col #adjust for start_col

        print("Center of rotation is at %0.2f px"%cor)  
        
        #create text file with centers of rotation
        if os.path.isfile(cfile): 
            with open(cfile) as f:
                modlines = []
                for line in f:
                    if os.path.basename(data_filename) in line:
                        modlines.append("%s %0.3f\n" %(os.path.basename(data_filename),cor))
                    else:
                        modlines.append(line)
                if not [k for k in modlines if os.path.basename(data_filename) in k]:
                    modlines.append("%s %0.3f\n" %(os.path.basename(data_filename),cor))                        
            with open(cfile,'w') as f:
                for line in modlines:
                    f.write(line)
        else:
            with open(cfile, 'w+') as f:
                f.write("%s %0.3f\n" %(os.path.basename(data_filename),cor))
                
        write_center_images(projection_data_full,theta,cor,cor,step_theta,path=file_path+'/AutoCenter_Vo_')
        
    if write_cor_images:
        print("Writing images for center of rotation check ...")        
        if find_center_automatically is False: 
            rot_guess = np.arange(center_guess - search_margin, center_guess + search_margin,search_step)
            guess_cor = center_guess
        else:
            rot_guess = np.arange(cor - search_margin, cor + search_margin,search_step)
            guess_cor = cor   
        rchunk = calculate_chunksize_cor(projection_data_full,len(rot_guess))
        print(rchunk)
        for i in range(1,len(rot_guess),rchunk):
            sim = i-1
            eim = i+rchunk-1
            if eim>len(rot_guess): 
                eim=len(rot_guess)
            rot_set = rot_guess[sim:eim]
            Parallel(n_jobs=ncore)(delayed(write_center_images)(projection_data_full,theta,guess,guess_cor,step_theta,path=file_path+'/DiagnoseCenter/img_') for guess in rot_set)
    del(projection_data_full)    
                    
### MAIN #######################################################################################################################

if __name__ == "__main__":
    start_time = time.time()
    all_folders =  next(os.walk(main_folder))[1]
    if data_key == 'ALL':
        subfold = [os.path.join(main_folder,j) for j in all_folders]
    else:
        subfold = [os.path.join(main_folder,j) for j in all_folders if data_key in j]
    for item in subfold:
        infold = next(os.walk(item))[1]
        done = [j for j in infold if 'DiagnoseCenter' in j]
        if done:
           print("Center has already been diagnosed for %s. Moving to next folder..."%os.path.basename(item))
           pass
        else:
            data_filename = glob.glob(item+ '/*Prefiltered.hdf5')[0]
            file_path = os.path.dirname(data_filename)+'/'
            bright_filename = None
            dark_filename = None
            #Give slice objects with the image ROI
            p,r,c = fget_dataset_shape(data_filename,row_slice= slice(None,None,1),column_slice=slice(None,None,1)) 
            if start_col is None:
               start_col = 0
            if end_col is None:
               end_col = c
            print("Starting center of rotation for %s" %os.path.basename(item))
            ftomopy_script()      
            print('Total elapsed time = %.1f min' % ((time.time() - start_time)/60.))

      

    
