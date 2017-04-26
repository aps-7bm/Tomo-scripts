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
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from joblib import Parallel, delayed
import tomopy 
import tomopy.misc.morph
import tomoPyScriptUtilities.BeamHardening as bh
from tomoPyScriptUtilities.reregistration import freregister
from tomoPyScriptUtilities.pad_elliptic import pad_elliptic


# USER SETTINGS ######################################################################################

# Lists of images to read
im_prefix="/home/SprayData/Cycle_2017_1/Tomography/Cummins/"
proj_filenames = glob.glob(im_prefix+"Cummins_04042017_Tomo_prefiltered_*.tif")
bright_images = glob.glob(im_prefix+"Cummins_04042017_Bright_prefiltered_*.tif")
bright_exposure_time_ratio = 1.0 # ratio of bright exposure to tomography exposure
dark_images = glob.glob(im_prefix+"*Cummins_04042017_Dark_prefiltered_*.tif")

# Recon settings
emission_mode = True # set True if you want to convert from transmission to path length of iron
pad_images = 'zero' # None, edge, zero, ellipse.  Ellipse is only designed for emission mode images.
remove_pad = 1.0 # remove fraction of padded region to make output files smaller?
pad_fraction_width = 0.5 # fraction of image width to pad on each side, if pad_images is not None
image_rescale = 1.0
algorithm = 'gridrec' # check tomopy docs for options
recon_filter = 'parzen' # check tomopy docs for options. this is for gridrec
phase_retrival = False

# Number of repeated images at each angle
n_repeat = 1

# Slices to reconstruct (crop image in the vertical)
smin_global=1140
smax_global=1150
sstep=1 # Phase retrieval requires step of 1

# Fraction of slices to load into RAM at once (smaller number reduces I/O overhead)
chunk=15 # Number of slices to reconstruct at once (RAM-limited)
margin=1 # Num slices at ends to not write (due to phase retrieval boundary effects)
nslices=1200 # Height of image / expected number of slices

# Center of rotation (None to scan & write diagnosis instead of reconstructing)
rotation_center=527

#Search pixel range for center of rotation.
diagnose_rotation_center = False
search_min=515
search_max=530

# Outlier threshold for zinger removal. Absolute intensity units.
zinger_level = 64500

# I0 region (for normalizing when doing conversion to path length)
I0_box_size=50 # pixels
I0_box_loc='lower right' # 'lower/upper/center left/right/middle'

# datum angle - 180-degree sweep starts at this value. Units radians
theta0 = 0 * np.pi/180. 
# Span of angles in reconstruction. Units radians
theta_span = 2 * np.pi

# Phase retreival parameters
z = 5 # propagation distance (cm)
eng = 62 # Flux-weighted mean energy of white beam (keV)
pxl = 5.86e-4 / 5. # 7BM PG camera with 5x objective
alpha = 1e-3 #6e-4 # Smaller number is smoother, larger number is less phase retrieval correction

# Where to write output
output_path_tiffstack = "./reconstruction_test_padded" # None for no TIFF output

#Bright field exposure time ratio
bright_exposure_time_ratio = 1.0

# Max number of CPUs for computation and I/O operations
ncore=16
ncore_io=4
ncore_phase=1

# END USER SETTINGS ###################################################################################

#If we are doing beam hardening, compute the polynomial function
if emission_mode:
    global coeffs
    coeffs = bh.fcompute_polynomial_function()
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
    print slice_M, slice_N
    box = data_subset[...,slice_M,slice_N]
    #plt.figure()
    #plt.imshow(box[0,...])
    #plt.colorbar()
    #Perform a mean across 
    if len(data_subset.shape) > 2:
        means = np.mean(box,axis=(1,2),dtype=np.float64)
    else:
        means = np.mean(box,dtype=np.float64)
    #plt.figure()    
    #plt.plot(means)
    #plt.show()
    #Return 1/mean values so we multiply to normalize, rather than divide.  Should be faster.
    return 1.0 / means

# Subroutine to read a set of n_repeat images starting at index i from proj_filenames list;
# Crop image, remove zingers, and average down stack to a flat image.
def fread_image_subset(proj_filenames,i,n_repeat,smin,smax,sstep=1,zinger_level=0.99,verbose=True,I0_inc=False,\
                       I0_box_size=200,I0_box_loc='lower left'):
    
    # Make sure to arrange data array correctly -- we want [angles, slices, transverse]
    data_subset = tomopy.io.reader.read_tiff_stack(proj_filenames[i], range(i,i+n_repeat), 5) # Load subset
    
    #Take the mean of the I0 region outside of the injector and save value as iO
    #This will be used to correct for any changes overall intensity   
    if I0_inc: i0=fbox_I0(data_subset,I0_box_size,I0_box_loc)

    if (smin<0):
        print 'Error: invalid minimum slice smin'; exit()
    elif (smax>data_subset.shape[1]):
        print 'Error: invalid maximum slice smax, array size is',data_subset.shape; exit()
    elif (smin is not None) and (smax is not None): # Crop
        if smax==-1: data_subset = data_subset[:,smin::sstep,:]
        else:        data_subset = data_subset[:,smin:smax:sstep,:]
    
    # If only one image in stack, skip filtering steps and go to I0
    if data_subset.shape[0] == 1:
        if I0_inc: return data_subset[0,...].astype(np.float32),i0
        else: return data_subset[0,...].astype(np.float32)    
    
    # Remove zingers before filtering
    med = np.median(data_subset,axis=0).astype(np.float32)
    zinger_counter = 0
    if zinger_level  < 1 :  zinger_level_intensity = zinger_level * np.nanmax(data_subset)
    else:  zinger_level_intensity = zinger_level  # If less than 1 assume float fraction value
    for j in range(n_repeat):
        slc=data_subset[j,...]
        zinger_counter += np.sum(slc>zinger_level_intensity)
        slc[slc>zinger_level_intensity] = med[slc>zinger_level_intensity]
        data_subset[j,...]=slc
        del slc
    del med
    print("Number of total zingers found = " + str(zinger_counter))
    
    # Mean filter
    flat = np.mean(data_subset,axis=0).astype(np.float32)
    
    if verbose:
            print "Images %i-%i" % (i,i+n_repeat),data_subset.shape,",",zinger_counter,\
            "zingers, output array size",flat.shape
            sys.stdout.flush()
    elif i==0:
            print "Each slice from %i images" % (n_repeat),data_subset.shape,\
                  "-> output array size",flat.shape        

    if I0_inc: return copy.deepcopy(flat),i0
    else: copy.deepcopy(flat)

def fread_images(image_filenames):
    '''Reads a set of images from disk.
    Input:
    image_filenames: names of all of the files to be read
    Output:
    numpy array with the image data.
    '''
    return tomopy.io.reader.read_tiff_stack(image_filenames[0],range(len(image_filenames)),5)


def fconvert_to_path_length_wrapper(proj):
    '''Converts the projection transmission data to pathlength of iron.
    Inputs:
    proj: numpy array of projection image data
    Output:
    numpy array of iron pathlength data.  Same shape as proj.
    '''
    global coeffs
    return np.polyval(coeffs,(np.log(proj)).astype(np.float32))

def fread_projection_images(image_filenames):
    '''Reads in the projection filenames.
    '''
    if len(proj_filenames)==0:
        print "Error: no images found"
        raise ValueError    
    # Loop through all images to load and crop
    t0=time.time()
    print("\n" + str(len(proj_filenames)) + " images detected.")
    print "\nReading data..."
    projection_data_full = tomopy.io.reader.read_tiff_stack(image_filenames[0],range(len(image_filenames)),5)
    print("\n The shape of the projection data is: " + str(projection_data_full.shape))
    print('\tElapsed time loading images: %.1f min\n' % ((time.time()-t0)/60.))
    return projection_data_full

def fread_bright_dark_images(bright_filenames,dark_filenames,projection_data):
    '''Loads the bright and dark images.
    '''    
    t0 = time.time()   
    # Load bright & dark fields, using median across the images if there are more than one.
    if bright_images is not None: 
        print "Reading Bright field images..."
        bright_all = tomopy.io.reader.read_tiff_stack(bright_filenames[0], range(len(bright_filenames)), 5)
        bright_median = np.median(bright_all,axis=0).astype(np.float32)
        bright_median /= float(bright_exposure_time_ratio)
        del(bright_all)
    else:
        bright_median = np.ones_like(projection_data[0,...])*np.max(projection_data)
    
    if dark_images is not None:
        print "Reading Dark field images..."
        dark_all = tomopy.io.reader.read_tiff_stack(dark_filenames[0], range(len(dark_filenames)), 5)
        #Mask off points that are far above the background level: maybe some readout weirdness?        
        dark_median = np.median(dark_all,axis=0).astype(np.float32)
        dark_mean = np.mean(dark_median)
        dark_std = np.std(dark_median)
        dark_median[dark_median > dark_mean + 4 * dark_std] = dark_mean
        del(dark_all)
    else:
        dark_median = np.zeros_like(projection_data[0,...])
    print('\tElapsed %i sec\n' % (time.time()-t0))
    return (bright_median, dark_median)

def fnorm_bright_dark(projection_data,bright_image,dark_image):
    '''Normalizes the projection data by the bright and dark fields.
    '''    
    print("Normalizing by bright and dark fields ...")    
    if len(projection_data.shape) == 2:
        return (projection_data - dark_image) / (bright_image - dark_image)
    #Stack of 2D images    
    else:
        return (projection_data - dark_image[None,...]) / (bright_image - dark_image)[None,...]

def fspecify_angles(projection_data):
    '''Computes the theta values for each image.
    Returns numpy array of angles.
    '''
    theta = np.linspace(0+theta0,theta_span+theta0,num=projection_data.shape[0])
    print("Data size: ",projection_data_full.shape," -- %i projections, %i slices, %i px wide" % projection_data_full.shape)
    return theta

def fprepare_images_tiff():
    '''Performs all initial readin steps for TIFF images.
    * Loads the projection images
    * Loads the bright and dark fields
    * Normalizes the projections by the bright and dark fields.
    * Computes the intensity of a background region and normalizes by it.
    '''
    # Import the projection images
    projection_data_full = fread_projection_images(proj_filenames)
    
    #Load the bright and dark images.
    bright_median, dark_median = fread_bright_dark_images(bright_images,dark_images,projection_data_full)
    
    #Normalize the projection images by the bright and dark fields
    projection_data_full = fnorm_bright_dark(projection_data_full,bright_median,dark_median)

    #Normalize by the I0 region
    #After this, images should be in terms of transmission.
    t0=time.time()
    print "Normalizing by I0 box ..."
    I0_means = fbox_I0(projection_data_full,I0_box_size,I0_box_loc)
    projection_data_full *= I0_means[:,None,None]
    print '\tElapsed %i sec\n' % (time.time()-t0)  

    # Specify angles
    '''theta = np.linspace(0+theta0,theta_span+theta0,num=projection_data_full.shape[0])
    print "Data size:",projection_data_full.shape," -- %i projections, %i slices, %i px wide" % projection_data_full.shape
    '''
    theta = fspecify_angles(projection_data_full)

    return (projection_data_full, theta)

#########################################################################################################################
# Main function to do tomoPy reconstruction. Input parameters are defined at the top of the script.
# This version will load ALL the images and work on chunks of specified size in RAM.
# Chunks are overlapped by some 'margin' to avoid edge phase retrieval errors.
def ftomopy_script_v2():
    
    # Import the projection images
    projection_data_full = fread_projection_images(proj_filenames)
    
    #plt.imshow(projection_data_full[0,...])
    #plt.colorbar()
    #plt.show()
    
    #Load the bright and dark images.
    bright_median, dark_median = fread_bright_dark_images(bright_images,dark_images,projection_data_full)
    ''' 
    # Load bright & dark fields, using median across the images if there are more than one.
    if bright_images is not None: 
        print "Reading Bright field images..."; sys.stdout.flush()
        bright_all = tomopy.io.reader.read_tiff_stack(bright_images[0], range(len(bright_images)), 5)
        bright_median = np.median(bright_all,axis=0).astype(np.float32)
        bright_median /= float(bright_exposure_time_ratio)
        del(bright_all)
    else:
        bright_median = np.ones_like(projection_data_full[0,...])*np.max(projection_data_full)
    
    if dark_images is not None:
        print "Reading Dark field images..."; sys.stdout.flush()
        dark_all = tomopy.io.reader.read_tiff_stack(dark_images[0], range(len(dark_images)), 5)
        dark_median = np.median(dark_all,axis=0).astype(np.float32)
        dark_mean = np.mean(dark_median)
        dark_std = np.std(dark_median)
        dark_median[dark_median > dark_mean + 4 * dark_std] = dark_mean
        del(dark_all)
    else:
        dark_median = np.zeros_like(projection_data_full[0,...])
    '''
    #plt.subplot(121)
    #plt.imshow(bright_median)
    #plt.colorbar()
    #plt.subplot(122)
    #plt.imshow(dark_median)
    #plt.colorbar()
    #plt.show()
        
    # Specify angles
    '''theta = np.linspace(0+theta0,theta_span+theta0,num=projection_data_full.shape[0])
    print "Data size:",projection_data_full.shape," -- %i projections, %i slices, %i px wide" % projection_data_full.shape
    '''
    theta = fspecify_angles(projection_data_full)

    #Normalize the projection images by the bright and dark fields
    '''
    t0=time.time()
    print "Normalizing by bright and dark fields ..."
    sys.stdout.flush()
    norm_denominator = bright_median - dark_median
    for i in range(projection_data_full.shape[0]):
        projection_data_full[i,:,:] = (projection_data_full[i,:,:] - dark_median) / norm_denominator
    #projection_data_full = tomopy.prep.normalize.normalize(projection_data_full,bright_median,dark_median)
    del(bright_median,dark_median)
    print '\tElapsed %i sec\n' % (time.time()-t0)
    #plt.imshow(projection_data_full[0,...])
    #plt.title('Before Norm')
    #plt.colorbar()'''
    projection_data_full = fnorm_bright_dark(projection_data_full,bright_median,dark_median)

    #Normalize by the I0 region
    #After this, images should be in terms of transmission.
    t0=time.time()
    print "Normalizing by I0 box ..."
    I0_means = fbox_I0(projection_data_full,I0_box_size,I0_box_loc)
    projection_data_full *= I0_means[:,None,None]
    print '\tElapsed %i sec\n' % (time.time()-t0)
    #plt.figure()
    #plt.imshow(projection_data_full[0,...])
    #plt.title('After Norm')
    #plt.colorbar()
    #plt.show()
    # Loop chunks: go from smin_global to smax_global
    for smin in np.arange(smin_global,smax_global,chunk-margin*2):
        tc=time.time()
        smax = smin + chunk
        if smax>projection_data_full.shape[1]: 
            smax=projection_data_full.shape[1]
        if (smax-2*margin)<=smin: 
            break
        print '\nWorking on slices %i to %i of image (%i slices total)' % (smin,\
                                                                   smax,projection_data_full.shape[1])

        #Extract chunk for further processing
        projection_data_subset = projection_data_full[:,smin:smax,:] # Extract chunk


        # Stripe removal
        t0=time.time()
        print "Removing stripes/rings..."; sys.stdout.flush()
        #data = tomopy.prep.stripe.remove_stripe_fw(data,level=8,wname='sym16',sigma=1,pad=True,ncore=ncore) # Fourier Wavelet method
        #data = tomopy.prep.stripe.remove_stripe_ti(data) # Titarenko algorithm
    	projection_data_subset = tomopy.prep.stripe.remove_stripe_sf(projection_data_subset, 10, ncore) # Smoothing filter stripe removal
        print '\tElapsed %i sec\n' % (time.time()-t0)

        # Phase retrieval
        if phase_retrival:
            print "Phase retrieval..."; sys.stdout.flush(); t0=time.time()
            projection_data_subset = tomopy.prep.phase.retrieve_phase(projection_data_subset,pixel_size=pxl,dist=z,energy=eng,alpha=alpha,\
                                            pad=True,ncore=ncore_phase)
            print '\tElapsed %i sec\n' % (time.time()-t0)

        #plt.imshow(projection_data_subset[0,...])
        #plt.colorbar()
        #plt.axis('tight')
        #plt.show()
        # Convert to path length of iron?        
        if emission_mode:
           t0=time.time()
           print "Converting units to path length of iron..."
           print "\tIn : mean=%.2f, min=%.2f, max=%.2f, std=%.2f" % (np.mean(projection_data_subset),
                                projection_data_subset.min(),projection_data_subset.max(),np.std(projection_data_subset))
           sys.stdout.flush()
           if ncore<=1:  # Serial processing (KM)
               projection_data_subset = fconvert_to_path_length_wrapper(projection_data_subset)
           else: # Parallel wrapper (DD)
               convertedList=Parallel(n_jobs=ncore,verbose=0)(delayed(fconvert_to_path_length_wrapper)(projection_data_subset[m,...].copy()) for m in range(projection_data_subset.shape[0]))
               for m in range(projection_data_subset.shape[0]): projection_data_subset[m,:,:] = convertedList[m].astype(np.float32)
           print "\tOut: mean=%.2f, min=%.2f, max=%.2f, std=%.2f" % (np.mean(projection_data_subset),
                                        projection_data_subset.min(),projection_data_subset.max(),np.std (projection_data_subset))
           print '\tElapsed %i sec\n' % (time.time()-t0)
        
        #plt.imshow(projection_data_subset[0,...])
        #plt.colorbar()
        #plt.axis('tight')
        #plt.show()          
        # Centering
        print "\nCenter of rotation..."; sys.stdout.flush()
        if diagnose_rotation_center:
            t0=time.time()
            print "Diagnosing center in horiz. range %i-%i px and saving samples..." % (search_min,search_max)
            tomopy.write_center(projection_data_subset,theta,dpath=output_path_tiffstack,\
                    cen_range=(search_min,search_max,0.5),ind=projection_data_subset.shape[1]/2,emission=emission_mode,mask=True,ratio=1.0)
            print "Wrote center test images."
            print '\tElapsed %i sec\n' % (time.time()-t0)
            print 'Total elapsed time = %.1f min' % ((time.time() - start_time)/60.)
            return
        
        #If we have > 180 degrees of projections, reregister
        if np.max(theta) - np.min(theta) > np.pi:
            print("Performing reregistration of full spin to a range 0-180 degrees.")
            working_data_subset, working_theta, working_rotation_center = freregister(projection_data_subset,theta,rotation_center)
        else:
            working_data_subset = projection_data_subset
            working_theta = theta
            working_rotation_center = rotation_center
        print("Rotation center = {:0.1f} pixels.".format(working_rotation_center))
        plt.plot(working_data_subset[0,0,:],'r.')
        plt.show()
        
        # Mask huge values, NaNs and INFs from the filtered data
        maxv=1e5
        working_data_subset[np.isnan(working_data_subset)]=0
        working_data_subset[np.isinf(working_data_subset)]=np.nanmax(working_data_subset)*np.sign(working_data_subset[np.isinf(working_data_subset)])
        working_data_subset[np.abs(working_data_subset)>maxv]=maxv*np.sign(working_data_subset[np.abs(working_data_subset)>maxv])
        
        # Pad the array
        if pad_images is not None:
            t0=time.time()
            prev_width = working_data_subset.shape[2]    
            print "\nPadding array...", working_data_subset.shape
            pad_pixels=int(np.ceil(prev_width*pad_fraction_width))
            if pad_images == 'zero':
                working_data_subset = np.pad(working_data_subset, ((0,0),(0,0),(pad_pixels,pad_pixels)),mode='constant', constant_values=0) # zero value pad
            elif pad_images == 'edge':
                working_data_subset = np.pad(working_data_subset, ((0,0),(0,0),(pad_pixels,pad_pixels)),mode='edge') # edge value pad
            elif pad_images == 'ellipse':
                # Autodetect radius from maximum LOS value-give scale in px/um
                working_data_subset = pad_elliptic(working_data_subset,center=center,radius=-1.0/(1e4*pxl),pad_amt=pad_pixels) 
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
        sys.stdout.flush()
        data_recon = tomopy.recon(working_data_subset * 0.8532, working_theta,center=working_rotation_center + center_offset,
                        emission=emission_mode,algorithm=algorithm,ncore=ncore,filter_name=recon_filter)
        print '\tElapsed %i sec\n' % (time.time()-t0)
               
        # Rearrange output (Dan notices that the images are flipped vertically!)
        data_recon = np.fliplr(data_recon) / image_rescale
    
        print "Chunk output stats: ", data_recon.shape
        print "Nans: ",np.sum(np.isnan(data_recon)),'/',np.product(data_recon.shape)
        print "Infs: ",np.sum(np.isinf(data_recon)),'/',np.product(data_recon.shape)
        #print "Mean/Std: ",np.mean(data_recon), np.std(data_recon) # Commented out because slow
        #print "Min/Max: ",np.nanmin(data_recon), np.nanmax(data_recon),"\n" # Commented out because slow
        
        # Mask NaNs and INFs from the output data.
        data_recon[np.isnan(data_recon)]=0
        data_recon[np.isinf(data_recon)]=np.nanmax(data_recon)*np.sign(data_recon[np.isinf(data_recon)])
        
        # If image is padded, throw away the extra voxels
        if (pad_images is not None) & (remove_pad>0):
            print "Cropping out the padded region: starting with volume size",data_recon.shape
            o=int(center_offset*remove_pad)
            data_recon=data_recon[:,o:-o+1,o:-o+1]
            print "Cropped volume size:",data_recon.shape,"\n"
        
        # Ring removal
        t0=time.time()
        print "\nRemoving rings..."; sys.stdout.flush()
        data_recon = tomopy.remove_ring(data_recon,ncore=ncore)
        print '\tElapsed %i sec\n' % (time.time()-t0)
        
        
        # Write 32-bit tiff stack out
        if output_path_tiffstack is not None:
            t0=time.time()
            print "Writing slices %i-%i to %s..." % (smin+margin, smin+margin+data_recon[margin:data_recon.shape[0]-margin+1,...].shape[0]-1,\
                output_path_tiffstack)
            sys.stdout.flush()
            
            tomopy.io.writer.write_tiff_stack(data_recon[margin:data_recon.shape[0]-margin+1,...],\
                                              axis=0,fname=output_path_tiffstack+'/img',\
                                              start=smin_global+smin+margin,\
                                              overwrite=True)
                         
            print '\tElapsed %i sec\n' % (time.time()-t0)

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
