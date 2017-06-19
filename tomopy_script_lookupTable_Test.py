#!/usr/bin/env python
"""
TomoPy emission tomography script for 7BM
Reads and writes in TIFF stack format (for easy viewing with ImageJ)
See: https://tomopy.readthedocs.org/en/latest/source/api.html

Tomopy needs the following additional modules: pywavelets, tifffile, matplotlib
My implementation for parallel file IO requires joblib

Daniel Duke
X-ray Fuel Spray
Energy Systems Division
Argonne National Laboratory

Started: Oct 14, 2015

Edits: Oct 21, 2015
       v2 now loads half the image slices to RAM and then splits this further into chunks,
       rather than loading the images again for every chunk. ftomopy_script_v2 is called
       multiple times, and each call will load a large number of slices s[min/max]_global
       and process them internally in 'chunk' sets of slices at once. Added HDF writer.

       Oct 28, 2015
       Fixed h5py writer and chunk slicing. Monitor memory usage between chunk iterations.
       Handling of n_repeat=1 for preprocessed data and bright field exposure time change.
       Added custom theta span option for partial reconstructions.

       Oct 29, 2015
       Fixed bug where very small number of slices might be reconstructed.

       Nov 16, 2015
       Updated code to allow for emission tomography reconstruction of images whose intensity
       has been corrected to path length via lookup table. This accounts for beam hardening.
       added flags to control emission/transmission mode, accept None for bright and dark
       images, add padding for emission mode, and image scale factor.

       Nov 18, 2015
       Added options to pad and change algorithm in emission tomography image.

       Jan 7, 2016
       Fixed emission mode padding, added ring removal and smoothing for tomopy-0.1.15

       Jan 8, 2016
       Ellipse padding feature added

       Feb 23, 2016
       Added some boolean flags so we don't have to comment code blocks as much.
       Faster sorting of images into the numpy array

       Feb 24, 2016
       Added more padding features: pad_fraction_width allows variable amount of padding,
       remove_pad allows choice of cropping output or writing full padded volume.
       Fixed angle linspace to allow for >180 degrees rotations.
       Tested with a 360 degree tomography: best settings were with edge-padding at 100% width.

       Feb 27, 2016
       Apply correction translations to fix precession of the axis of rotation (fcorrect_precession). 
       Assumed uniform across image height for now.

       Feb 28, 2016
       Automatic detection of precession correction CSV file.

       Apr 27, 2016
       Included path length of iron conversion using lookup table from ALK. I0 is calculated while images
       are being read. "emission_mode=True" flag now turns on use of lookup table rather than assuming
       input images are in path length units. Dan added a parallelization wrapper for this feature to speed
       things up. Cleaned the code up a bit.

       Apr 28, 2016
       Fixed slicing of bright and dark fields depending on whether arrays are 2D or 3D by using ellipsis on
       line 361. Added middle and center options for the I0 box.

       Apr 29, 2016
       Added reregistration function for emission tomography when doing spins > 180 degrees. TomoPy can only
       natively handle >180 degree spin reregistration when in transmission tomography mode.
"""

import time, sys, copy, glob, gc
import os
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from joblib import Parallel, delayed
import multiprocessing
import dxchange.writer
import tomopy.recon.rotation
import tomopy 
import tomopy.misc.morph
import tomopy.prep
import tomoPyScriptUtilities.BeamHardening_Transmission as bh
from tomoPyScriptUtilities.reregistration import freregister
from tomoPyScriptUtilities.pad_elliptic import pad_elliptic


# USER SETTINGS ######################################################################################

# Lists of images to read
file_path = "/home/akastengren/data/Cycle_2017_1/2015_Test_01/"
file_root = 'Gerardi_2015_Test_01_04202017'
#im_prefix="/home/akastengren/data/SprayData/Cycle_2017_1/Tomography/Gerardi/115um_J/"
# proj_filenames = glob.glob(im_prefix+"Cummins_04042017_Tomo_prefiltered_*.tif")
# bright_images = glob.glob(im_prefix+"Cummins_04042017_Bright_prefiltered_*.tif")
bright_exposure_time_ratio = 1.0 # ratio of bright exposure to tomography exposure
# dark_images = glob.glob(im_prefix+"*Cummins_04042017_Dark_prefiltered_*.tif")

# Recon settings
emission_mode = True # set True if you want to convert from transmission to path length of iron
pad_images = 'zero' # None, edge, zero, ellipse.  Ellipse is only designed for emission mode images.
remove_pad = 1.0 # remove fraction of padded region to make output files smaller?
pad_fraction_width = 0.5 # fraction of image width to pad on each side, if pad_images is not None
image_rescale = 1.0
algorithm = 'gridrec' # check tomopy docs for options
recon_filter = 'parzen' # check tomopy docs for options. this is for gridrec
phase_retrieval = False

# Number of repeated images at each angle
n_repeat = 1

# Slices to reconstruct (crop image in the vertical)
smin_global=1
smax_global=1199
sstep=1 # Phase retrieval requires step of 1

# Fraction of slices to load into RAM at once (smaller number reduces I/O overhead)
chunk=50 # Number of slices to reconstruct at once (RAM-limited)
margin=1 # Num slices at ends to not write (due to phase retrieval boundary effects)
nslices=1200 # Height of image / expected number of slices

# Center of rotation (None to scan & write diagnosis instead of reconstructing)
rotation_center=933.5

#Search pixel range for center of rotation.
diagnose_rotation_center = False
search_min=315
search_max=530

# Outlier threshold for zinger removal. Absolute intensity units.
zinger_level = 64500

# I0 region (for normalizing when doing conversion to path length)
I0_box_size=50 # pixels
I0_box_loc='upper right' # 'lower/upper/center left/right/middle'

# datum angle - 180-degree sweep starts at this value. Units radians
theta0 = 0 * np.pi/180. 
# Span of angles in reconstruction. Units radians
theta_span = 1.0 * np.pi

# Phase retreival parameters
z = 5 # propagation distance (cm)
eng = 62 # Flux-weighted mean energy of white beam (keV)
pxl = 5.86e-4 / 5. # 7BM PG camera with 5x objective
alpha = 1e-3 #6e-4 # Smaller number is smoother, larger number is less phase retrieval correction

# Where to write output
output_path_tiffstack = file_path +"reconstruction_test_padded/" # None for no TIFF output

#Bright field exposure time ratio
bright_exposure_time_ratio = 1.0

# Max number of CPUs for computation and I/O operations
ncore=multiprocessing.cpu_count()
ncore_io=4
ncore_phase=1

# END USER SETTINGS ###################################################################################

#Handle beam hardening
filters = {bh.Be_filter:750.0,bh.Mo_filter:25.0,bh.Cu_filter:250.0}
filtered_spectrum = bh.fapply_filters(filters)
# plt.plot(bh.spectral_energies,filtered_spectrum)
# plt.show()
global coeffs
coeffs = bh.fcompute_lookup_coeffs(bh.spectral_energies,filtered_spectrum)
print("\nCoefficients for fit for transmission to pathlength conversion complete.")

def fbox_I0(data_subset,I0_box_size,I0_box_loc):
    '''Extracts I0 region and provides average of this region for each image.
    Input: 
    data_subset: set of images across which we want I0 data
    I0_box_size, I0_box_loc: size and location of I0 box
    Output:
    Array same length as the first dimension of data_subset with mean value for each image
    '''
    if len(data_subset.shape)>2:
        M = data_subset.shape[1]
        N = data_subset.shape[2]
    else:
        M = data_subset.shape[0]
        N = data_subset.shape[1]
    mid_row = int((M+I0_box_size)/2.)
    mid_col = int((N+I0_box_size)/2.)
    #Make the location string lower case and split it
    I0_box_split=I0_box_loc.lower().split()
    #Figure out the vertical location
    slice_M = None
    slice_N = None
    if I0_box_split[0] == 'lower':
        slice_M = slice(-I0_box_size,None,1)
    elif I0_box_split[0] == 'upper':
        slice_M = slice(0,I0_box_size,1)
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
    #Extract the subset
    box = data_subset[...,slice_M,slice_N]
    #Perform a mean across 
    if len(data_subset.shape) > 2:
        means = np.mean(box,axis=(1,2),dtype=np.float32)
    else:
        means = np.mean(box,dtype=np.float32)
    #Return 1/mean values so we multiply to normalize, rather than divide.  Should be faster.
    return 1.0 / means
# 
# def fread_images(image_filenames):
#     '''Reads a set of images from disk.
#     Input:
#     image_filenames: names of all of the files to be read
#     Output:
#     numpy array with the image data.
#     '''
#     return tomopy.io.reader.read_tiff_stack(image_filenames[0],range(len(image_filenames)),5)


def fconvert_to_mono_wrapper(proj):
    '''Converts the projection transmission data to pathlength of iron.
    Inputs:
    proj: numpy array of projection image data
    Output:
    numpy array of projections rescaled to mono beam.  Same shape as proj.
    '''
    return np.polyval(coeffs,(np.log(proj)).astype(np.float32))

# def fread_projection_images(image_filenames):
#     '''Reads in the projection filenames.
#     '''
#     if len(proj_filenames)==0:
#         print "Error: no images found"
#         raise ValueError    
#     # Loop through all images to load and crop
#     t0=time.time()
#     print("\n" + str(len(proj_filenames)) + " images detected.")
#     print "\nReading data..."
#     projection_data_full = tomopy.io.reader.read_tiff_stack(image_filenames[0],range(len(image_filenames)),5)
#     print("\n The shape of the projection data is: " + str(projection_data_full.shape))
#     print('\tElapsed time loading images: %.1f min\n' % ((time.time()-t0)/60.))
#     return projection_data_full

# def fread_bright_dark_images(bright_filenames,dark_filenames,projection_data):
#     '''Loads the bright and dark images.
#     '''    
#     t0 = time.time()   
#     # Load bright & dark fields, using median across the images if there are more than one.
#     if bright_images is not None: 
#         print "Reading Bright field images..."
#         bright_all = tomopy.io.reader.read_tiff_stack(bright_filenames[0], range(len(bright_filenames)), 5)
#         bright_median = np.median(bright_all,axis=0).astype(np.float32)
#         bright_median /= float(bright_exposure_time_ratio)
#         del(bright_all)
#     else:
#         bright_median = np.ones_like(projection_data[0,...])*np.max(projection_data)
#     
#     if dark_images is not None:
#         print "Reading Dark field images..."
#         dark_all = tomopy.io.reader.read_tiff_stack(dark_filenames[0], range(len(dark_filenames)), 5)
#         #Mask off points that are far above the background level: maybe some readout weirdness?        
#         dark_median = np.median(dark_all,axis=0).astype(np.float32)
#         dark_mean = np.mean(dark_median)
#         dark_std = np.std(dark_median)
#         dark_median[dark_median > dark_mean + 4 * dark_std] = dark_mean
#         del(dark_all)
#     else:
#         dark_median = np.zeros_like(projection_data[0,...])
#     print('\tElapsed %i sec\n' % (time.time()-t0))
#     return (bright_median, dark_median)

def fread_projection_images_fly_2017_2():
    '''Loads the bright and dark images.
    '''    
    t0 = time.time()   
    data_filename = file_path + file_root + "_Prefiltered.hdf5"
    with h5py.File(data_filename,'r') as data_hdf:
        data = data_hdf['Prefiltered_images'][:,1:,:]
    print('\tElapsed reading data: %i sec\n' % (time.time()-t0))
    return data.astype(np.float32)

def fread_bright_dark_images_fly_2017_2():
    '''Loads the bright and dark images.
    '''    
    t0 = time.time()   
    bright_filename = file_path + file_root + "_Bright_Prefiltered.hdf5"
    with h5py.File(bright_filename,'r') as bright_hdf:
        bright_data = bright_hdf['Prefiltered_images'][1:,:]
    dark_filename = file_path + file_root + "_Dark_Prefiltered.hdf5"
    with h5py.File(dark_filename,'r') as dark_hdf:
        dark_data = dark_hdf['Prefiltered_images'][1:,:]
    print('\tElapsed reading bright/dark: %i sec\n' % (time.time()-t0))
    return (bright_data.astype(np.float32), dark_data.astype(np.float32))

def fnorm_bright_dark(projection_data,bright_image,dark_image):
    '''Normalizes the projection data by the bright and dark fields.
    '''    
    t0 = time.time()
    print("Normalizing by bright and dark fields ...") 
    denominator = bright_image - dark_image   
    if len(projection_data.shape) == 2:
        return (projection_data - dark_image) / denominator
    #Stack of 2D images    
    else:
        for i in range(projection_data.shape[0]):
            projection_data[i,...] = (projection_data[i,...] - dark_image) / denominator
        print("\tElapsed in bright/dark corrections: {0:6.3f} sec".format(time.time() - t0))
        return projection_data

def fspecify_angles(projection_data):
    '''Computes the theta values for each image.
    Returns numpy array of angles.
    '''
    theta = np.linspace(0+theta0,theta_span+theta0,num=projection_data.shape[0])
    print("Data size: ",projection_data.shape," -- %i projections, %i slices, %i px wide" % projection_data.shape)
    return theta

def fnormalize_image_brightness(projection_data):
    '''Uses a small region outside the sample to normalize for shifts
    in image intensity.
    '''
    t0=time.time()
    print("Normalizing by I0 box ...")
    I0_means = fbox_I0(projection_data,I0_box_size,I0_box_loc)
    projection_data *= I0_means[:,None,None]
    print("\tElapsed normalizing image by I0: {0:6.3f} sec".format(time.time() - t0))
    return projection_data

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
        dataset = tomopy.prep.stripe.remove_stripe_sf(dataset, 10, ncore=1) # Smoothing filter stripe removal
    print("\tElapsed removing chunk stripes: {0:6.3f} sec".format(time.time() - t0))
    return dataset

def fbeam_hardening_function(coeffs,dataset):
    return (np.polyval(coeffs,np.log(dataset)).astype(np.float32))

def fapply_beam_hardening_projection(dataset):
    '''Applies beam hardening corrections.
    Uses joblib to speed this up.  
    Send bigger chunks to joblib so we have less process overhead.  
    If we send in smaller chunks, it doesn't work as well.
    '''
    t0=time.time()
    print("Apply beam hardening...")
    joblib_chunk = int(np.ceil(float(dataset.shape[1])/float(multiprocessing.cpu_count() - 1)))
    slices = []
    for i in range(0,dataset.shape[1],joblib_chunk):
        slices.append(slice(i,i+joblib_chunk,1))
#     print slices
    result_list = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(fbeam_hardening_function)(coeffs,dataset[:,s,:].copy()) for s in slices)
    for i,s in enumerate(slices):
        dataset[:,s,:] = result_list[i]
    print("\tElapsed in beam hardening: {0:6.3f} sec".format(time.time() - t0))
    return dataset

def fapply_phase_retrieval(dataset):
    '''Applies beam hardening corrections.
    '''
    t0=time.time()
    print("Apply phase retrieval...")
    dataset = tomopy.prep.phase.retrieve_phase(dataset,pixel_size=pxl,dist=z,energy=eng,alpha=alpha,\
                                            pad=True,ncore=ncore_phase)
    print("\tElapsed in phase retrieval: {0:6.3f} sec".format(time.time() - t0))
    return dataset

def fdiagnose_rotation_center(dataset,theta):
    '''Find the rotation center.
    '''
    #Try an automated way
    new_rotation_center = tomopy.find_center_pc(dataset[0,...], dataset[-1,...], tol=0.5)
    print(new_rotation_center)
    rotation_center = int(new_rotation_center)
    #Try the automated way
    new_rotation_center = tomopy.find_center(dataset, theta, ind=search_min, init=rotation_center, tol=0.5,ratio = 0.05)
    print(new_rotation_center)
    
    #Try writing images, as a back check
    tomopy.write_center(dataset, theta, dpath=file_path, 
                                       cen_range=[rotation_center-10,rotation_center+10,0.5], ind=search_min)
    return new_rotation_center

def freregister_data(dataset,theta,rotation_center):
    #If we have > 180 degrees of projections, reregister
    if np.max(theta) - np.min(theta) > np.pi:
        t0=time.time()
        print("\nRe-registering projections for span > 180 degrees...")
    
        # Find starting point for angles > 180 degrees
        #Account for roundoff error
        delta_theta = (theta[-1] - theta[0]) / (float(theta.shape[0]) - 1.0)
        index_180 = np.argmax(theta >= (np.pi - delta_theta / 10.0)) #Returns first argument where this is True
        # Make new theta array
        new_theta = theta[:index_180]
        #Verify that the angles before and after 180 degrees match closely
        if (np.allclose(new_theta+np.pi,theta[index_180:2*index_180])):
            print("Angles before and after 180 degrees match.")
        else:
            raise ValueError()
        #Round the center of rotation to the nearest half pixel.
        print('Initial center of rotation = {0:.1f}'.format(rotation_center))
        rotation_center = np.round(rotation_center * 2.0) / 2.0
        
        # Find new width of array
        old_image_center = (dataset.shape[2] - 1.0) / 2.0      #In between middle two pixels
        overlap_width = dataset.shape[2]  - np.abs(old_image_center - rotation_center)
        new_width = dataset.shape[2]  + int(2.0 * np.abs(old_image_center - rotation_center))
        new_center = float(new_width - 1) / 2.0
        print("Overlap between views 180 degrees apart = {0:0.1f} pixels.".format(overlap_width))
        print("New image width = {0:0.1f} pixels.".format(new_width))
        print("New image center = {0:0.1f} pixels.".format(new_center))
    
        # Make new dataset array 
        print "\tExisting data shape=",dataset.shape
        new_shape=(index_180,dataset.shape[1],new_width)
        print "\tRe-registering projections into a new array of shape = ",new_shape
        data_rereg = np.empty(new_shape,dtype=dataset.dtype) # Save the time of filling with zeros vs. np.zeros!
    
        # Figure out mapping of old data to the new array
        #Determine if the rotation axis is to the left or to the right of the center of the image
        if rotation_center >= old_image_center:
            old_image_slice = slice(0,int(np.ceil(rotation_center-0.01)),1)
            new_image_slice_0_180 = slice(0,int(np.ceil(new_center-0.01)),1)
            new_image_slice_180_360 = slice(data_rereg.shape[2]-1,int(np.floor(rotation_center + 0.01)),-1)
        else:
            old_image_slice = slice(int(np.ceil(rotation_center-0.01)),dataset.shape[2],1)
            new_image_slice_0_180 = slice(int(np.ceil(new_center-0.01)),data_rereg.shape[2],1)
            new_image_slice_180_360 = slice(int(np.floor(new_center+0.01)),None,-1)
        data_rereg[...,new_image_slice_0_180]=dataset[:index_180,:,old_image_slice]
        data_rereg[...,new_image_slice_180_360]=dataset[index_180:2*index_180,:,old_image_slice]
        return(data_rereg,new_theta,new_center)
    else:
        print("No reregistration of angles needed.")
        return(dataset,theta,rotation_center)
        

#########################################################################################################################
# Main function to do tomoPy reconstruction. Input parameters are defined at the top of the script.
# This version will load ALL the images and work on chunks of specified size in RAM.
# Chunks are overlapped by some 'margin' to avoid edge phase retrieval errors.
def ftomopy_script_v2():
    
    # Import the projection images
#     projection_data_full = fread_projection_images(proj_filenames)
    print("TomoPy version = " + tomopy.__version__)
    projection_data_full = fread_projection_images_fly_2017_2()
    plt.figure()
    plt.imshow(projection_data_full[20,...])
    plt.title('Before Norm')
    plt.colorbar()
    #Load the bright and dark images.
    #bright_median, dark_median = fread_bright_dark_images(bright_images,dark_images,projection_data_full)
    bright_median, dark_median = fread_bright_dark_images_fly_2017_2()
        
    # Specify angles
    '''theta = np.linspace(0+theta0,theta_span+theta0,num=projection_data_full.shape[0])
    print "Data size:",projection_data_full.shape," -- %i projections, %i slices, %i px wide" % projection_data_full.shape
    '''
    theta = fspecify_angles(projection_data_full)

    #Normalize the projection images by the bright and dark fields
    projection_data_full = fnorm_bright_dark(projection_data_full,bright_median,dark_median)

    #Normalize by the I0 region
    #After this, images should be in terms of transmission.
    projection_data_full = fnormalize_image_brightness(projection_data_full)
    plt.figure()
    plt.imshow(projection_data_full[0,...])
    plt.title('After Norm')
    plt.colorbar()
    
    #Check centering, if desired.
    if diagnose_rotation_center:
        fdiagnose_rotation_center(projection_data_full,theta)
        plt.show()
        return    
    
    ta = time.time()
    # Loop chunks: go from smin_global to smax_global
    for smin in np.arange(smin_global,smax_global,chunk):
        tc=time.time()
        smax = smin + chunk
        if smax>projection_data_full.shape[1]: 
            smax=projection_data_full.shape[1]
        print('\nWorking on slices {0:d} to {1:d} of image ({2:d} slices total)'.format(smin,\
                                                                   smax,projection_data_full.shape[1]))
 
        #Extract chunk for further processing
        projection_data_subset = projection_data_full[:,smin:smax,:] # Extract chunk
  
        # Stripe removal
        projection_data_subset = fremove_stripes(projection_data_subset)
        
        #Correct for beam hardening.  Now in terms of pathlength
        projection_data_subset = fapply_beam_hardening_projection(projection_data_subset)
        
        #If desired, perform phase retrieval
        if phase_retrieval:
            projection_data_subset = fapply_phase_retrieval(projection_data_subset)
            
#         projection_data_full[:,smin:smax,:] = projection_data_subset
        print('\tTime elapsed prepping chunk {0:6.3f} s.'.format(time.time() - tc))
        
        #Take care of datasets where we do a 0-360 degree rotation to increase FOV.
        working_data_subset, working_theta, working_rotation_center = freregister_data(projection_data_subset,theta,rotation_center)
        print("Rotation center = {:0.1f} pixels.".format(working_rotation_center))
        
        # Mask huge values, NaNs and INFs from the filtered data
        maxv=1e5
        working_data_subset[np.isnan(working_data_subset)]=0
        working_data_subset[working_data_subset <= 0] = 0
        working_data_subset[np.isinf(working_data_subset)]=np.nanmax(working_data_subset)*np.sign(working_data_subset[np.isinf(working_data_subset)])
        working_data_subset[np.abs(working_data_subset)>maxv]=maxv*np.sign(working_data_subset[np.abs(working_data_subset)>maxv])
         
        # Pad the array
        if pad_images is not None:
            t0=time.time()
            prev_width = working_data_subset.shape[2]    
            print "\nPadding array...", working_data_subset.shape
            pad_pixels=int(np.ceil(prev_width*pad_fraction_width))
            if pad_images == 'zero':
                working_data_subset = np.pad(working_data_subset, ((0,0),(0,0),(pad_pixels,pad_pixels)),mode='constant', constant_values=1.0) # zero value pad
            elif pad_images == 'edge':
                working_data_subset = np.pad(working_data_subset, ((0,0),(0,0),(pad_pixels,pad_pixels)),mode='edge') # edge value pad
#             elif pad_images == 'ellipse':
#                 # Autodetect radius from maximum LOS value-give scale in px/um
#                 working_data_subset = pad_elliptic(working_data_subset,center=center,radius=-1.0/(1e4*pxl),pad_amt=pad_pixels) 
            else:
                raise ValueError("pad_images = "+pad_images+" not understood!")
             
            print "\tNew size:",working_data_subset.shape
            print '\tElapsed %i sec\n' % (time.time()-t0)
            # Center offset
            center_offset = (working_data_subset.shape[2]-prev_width)/2
        else:
            center_offset=0
 
        # Tomographic reconstruction
        t0=time.time()
        print "\nReconstructing..."
        data_recon = tomopy.recon(working_data_subset, working_theta,center=working_rotation_center + center_offset,
                        algorithm=algorithm,ncore=ncore,filter_name=recon_filter)
        print '\tElapsed %i sec\n' % (time.time()-t0)
                
        # Rearrange output (Dan notices that the images are flipped vertically!)
        data_recon = np.fliplr(data_recon) / image_rescale
     
        print "Chunk output stats: ", data_recon.shape
        print "Nans: ",np.sum(np.isnan(data_recon)),'/',np.product(data_recon.shape)
        print "Infs: ",np.sum(np.isinf(data_recon)),'/',np.product(data_recon.shape)
         
        # Mask NaNs and INFs from the output data.
        data_recon[np.isnan(data_recon)]=0
        data_recon[np.isinf(data_recon)]=np.nanmax(data_recon)*np.sign(data_recon[np.isinf(data_recon)])
         
        # If image is padded, throw away the extra voxels
        if (pad_images is not None) & (remove_pad>0):
            print "Cropping out the padded region: starting with volume size",data_recon.shape
            o=int(center_offset*remove_pad)
            data_recon=data_recon[:,o:-o+1,o:-o+1]
            print "Cropped volume size:",data_recon.shape,"\n"
         
        #Ring removal
#         t0=time.time()
#         print "\nRemoving rings..."
#         data_recon = tomopy.remove_ring(data_recon.copy(),ncore=1)
#         print '\tElapsed %i sec\n' % (time.time()-t0)
         
         
        # Write 32-bit tiff stack out
        if output_path_tiffstack is not None:
            t0=time.time()
            print "Writing slices %i-%i to %s..." % (smin, smax,\
                output_path_tiffstack)
            for i in range(data_recon.shape[0]):
                dxchange.write_tiff(data_recon[i,...],\
                                    fname=output_path_tiffstack+'img_'+str(smin+i),overwrite=True)
                          
            print '\tElapsed %i sec\n' % (time.time()-t0)
        
        del(data_recon)
        del(projection_data_subset)
        del(working_data_subset)

    print('\tTotal Elapsed {0:6.3f} sec\n'.format(time.time()-ta))
    plt.figure()
    plt.imshow(projection_data_full[0,...])
    plt.title('After Beam Hardening')
    plt.colorbar()
    plt.show()

#         

#         
    return





### MAIN #######################################################################################################################

if __name__ == "__main__":
    start_time = time.time()
    ftomopy_script_v2()
    '''
    # Do center of rotation check
    if diagnose_rotation_center:
        ftomopy_script_v2()
        
            ftomopy_script_v2(proj_filenames, bright_images, dark_images, n_repeat, bright_exposure_time_ratio,\
                              smin_global, smax_global, sstep, chunk, margin, None, zinger_level, theta0, theta_span,\
                              z, eng, pxl, alpha, output_path_tiffstack, ncore, ncore_io, image_rescale, emission_mode,\
                              pad_images, algorithm, recon_filter, remove_pad, pad_fraction_width,\
                              I0_box_size,I0_box_loc)
            
        exit()
    
    
    # Due to RAM limit, reconstruct slices in parts so we are only holding part of the data in 
    # RAM at once. Make chunk_global as small as possible to reduce I/O overhead.
    smax_global0 = smax_global
    for smin_global in np.arange(chunk_global)*(nslices/chunk_global)+smin_global:
        smax = smin_global+2*margin+(nslices/chunk_global) # allow overlap for margin
        if (smax > smax_global0) & (smax_global0>=0): smax=smax_global0
        smax_global=smax
        if smax_global > nslices: smax_global=nslices
        
        print '\n--- %i/%i: Loading slices %i-%i of %i into RAM ---\n' % \
             (1+smin_global/(nslices/chunk_global),chunk_global,smin_global,smax_global,nslices) #(smax_global-smin_global+1))
        
        ftomopy_script_v2(proj_filenames, bright_images, dark_images, n_repeat, bright_exposure_time_ratio,\
                          smin_global, smax_global, sstep, chunk, margin, center, zinger_level, theta0,\
                          theta_span, z, eng, pxl, alpha, output_path_tiffstack, ncore, ncore_io, image_rescale,\
                          emission_mode, pad_images, algorithm, recon_filter, remove_pad, pad_fraction_width,\
                          I0_box_size,I0_box_loc)
         
        gc.collect()'''

    print 'Total elapsed time = %.1f min' % ((time.time() - start_time)/60.)
