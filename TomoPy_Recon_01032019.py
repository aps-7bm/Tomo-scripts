#!/usr/bin/env python
"""
Code developed in Cycle 2018-3 to feature additional attributes that are being now saved to the HDF5 file. 
The goal of the script is to input everything that the reconstruction would need to run successfully 
with as little user input as possible. This code also inputs the file name and center of rotation from a text file.
"""
#from PIL import Image
import time
import os
import copy
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
from joblib import Parallel, delayed
import multiprocessing
import dxchange.writer
import tomopy 
#import tomopy.misc.morph
#import tomopy.prep
import tomoPyScriptUtilities.BeamHardening_Transmission as bh
import sys
import psutil
#from skimage.feature import register_translation

# USER SETTINGS ######################################################################################

# Lists of images to read
centers_file = "/mnt/SprayData/Cycle_2018_3/Tomography/Cycle2018_3_Centers.txt"
data_key = 'ALL' #write ALL if all datasets should be reconstructed
bright_filename = None
dark_filename = None

# Recon settings
pad_images = 'edge' #None, zero, edge
pad_fraction_width = 0.8 # fraction of image width to pad on each side, if pad_images is not None
remove_pad = 1.0 # remove fraction of padded region to make output files smaller?

algorithm = 'gridrec' # check tomopy docs for options
recon_filter = 'parzen' # check tomopy docs for options. this is for gridrec
phase_retrieval = False

#crop image in horizontal - None if no cropping required
start_col = None
end_col = None

#I0 region (for normalizing when doing conversion to path length)
I0_normalize = True
I0_box_size = 40 # pixels 
I0_box_loc = 'center left' # 'lower/upper/center left/right/middle'. Skips first two rows due to bad pixel values.

# Phase retreival parameters
z = 5 # propagation distance (cm)
eng = 62 # Flux-weighted mean energy of white beam (keV)
alpha = 1e-3 #6e-4 # Smaller number is smoother, larger number is less phase retrieval correction

# Max number of CPUs for computation and I/O operations
ncore=multiprocessing.cpu_count()-1

#Transmission mode
beam_hardening = True
scint_material = bh.LuAG_scint #either LuAG, YAG, or LYSO
scintillator_thickness = 100.0
sample_material = bh.Si_filter
magnification = 10
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

#Calculate beam hardening correction coefficients from input filter values
def fbeam_hardening_function(coeffs,dataset):
    '''Return data in pathlength of pixels.
    '''
    return (np.polyval(coeffs,np.log(dataset)).astype(np.float32))

#Apply beam hardening correction
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
    result_list = Parallel(n_jobs=ncore)(delayed(fbeam_hardening_function)(coeffs,dataset[:,s,:].copy()) for s in slices)
    for i,s in enumerate(slices):
        dataset[:,s,:] = result_list[i]
    print(("\tElapsed in beam hardening: {0:6.3f} sec".format(time.time() - t0)))
    return dataset

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
        dataset = tomopy.prep.stripe.remove_stripe_sf(dataset, 10, ncore=1) # Smoothing filter stripe removal
    print(("\tElapsed removing chunk stripes: {0:6.3f} sec".format(time.time() - t0)))
    return dataset

#phase retrieval algorithm, if using
def fapply_phase_retrieval(dataset):
    t0=time.time()
    print("Apply phase retrieval...")
    dataset = tomopy.prep.phase.retrieve_phase(dataset,pixel_size=pixel_size,dist=z,energy=eng,alpha=alpha,\
                                            pad=True,ncore=ncore)
    print(("\tElapsed in phase retrieval: {0:6.3f} sec".format(time.time() - t0)))
    return dataset

#re-register data for 360 deg scans onto 0-180 deg
def freregister_data(dataset,theta,rotation_center,center_location):
    #If we have > 180 degrees of projections, reregister
    if np.max(theta) - np.min(theta) > np.pi:
        print("\nRe-registering projections for span > 180 degrees...")    
        # Find starting point for angles > 180 degrees, account for roundoff error
        delta_theta = step_theta
        index_180 = np.argmax(theta >= (np.pi - delta_theta / 10.0)) #Returns first argument where this is True
        # Make new theta array
        new_theta = theta[:index_180]
        #Verify that the angles before and after 180 degrees match closely
        #if (np.allclose(new_theta+np.pi,theta[index_180:2*index_180])):
        #    print("Angles before and after 180 degrees match.")
        #else:
        #    raise ValueError()
        #Round the center of rotation to the nearest half pixel.
        rotation_center = np.round(rotation_center * 2.0) / 2.0
        #print('Initial center of rotation = %0.2f' %rotation_center)        
        # Find new width of array
        old_image_center = (dataset.shape[2] - 1.0) / 2.0      #In between middle two pixels
        overlap_width = float(dataset.shape[2])  - 2.0* np.abs(old_image_center - rotation_center)
        new_width = dataset.shape[2]  + int(2.0 * np.abs(old_image_center - rotation_center))
        new_center = float(new_width - 1) / 2.0
        #print(("Overlap between views 180 degrees apart = {0:0.1f} pixels.".format(overlap_width)))
        #print(("New image width = {0:0.1f} pixels.".format(new_width)))
        #print(("New image center = {0:0.1f} pixels.".format(new_center)))    
        # Make new dataset array 
        data_rereg = tomopy.sino_360_t0_180(dataset,overlap_width,center_location)
        return(data_rereg,new_theta,new_center)
    else:
        print("No reregistration of angles needed.")
        return(dataset,theta,rotation_center)

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
    # Center offset
    center_offset = (padded_dataset.shape[2]-dataset.shape[2])/2
    return (padded_dataset,center_value + center_offset)

#write out the tiffstack of reconstructed images
def fwrite_tiff_stack(recon_dataset,smin,total_rows):
     # Write 32-bit tiff stack out
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
    # Tomographic reconstruction
    t0=time.time()
    print("\nReconstructing...")
    data_recon = tomopy.recon(dataset, theta,center=center,
                    algorithm=algorithm,ncore=ncore,filter_name=recon_filter)
    print('\tElapsed %i sec\n' % (time.time()-t0))
            
    #Crop the image to the desired size
    #print(("Cropping volume from size {0:d} to size {1:d}".format(data_recon.shape[1],final_size)))
    crop_margin = int((data_recon.shape[1] - final_size) / 2.0)
    if crop_margin > 0:
        data_recon = data_recon[:,crop_margin:crop_margin+final_size,crop_margin:crop_margin+final_size]

    # Rearrange output (Dan notices that the images are flipped vertically!)
    data_recon = np.fliplr(data_recon)
 
    #print(("Chunk output stats: ", data_recon.shape))
    #print(("Nans: ",np.sum(np.isnan(data_recon)),'/',np.product(data_recon.shape)))
    #print(("Infs: ",np.sum(np.isinf(data_recon)),'/',np.product(data_recon.shape)))
     
    # Mask NaNs and INFs from the output data.
    data_recon[np.isnan(data_recon)]=0
    data_recon[np.isinf(data_recon)]=np.nanmax(data_recon)*np.sign(data_recon[np.isinf(data_recon)])
    print(('\tTotal elapsed to reconstruct subset = {0:3.3f} s.'.format(time.time() - t0)))
    return data_recon

#remove outliers with a median filter      
def remove_outliers(data,dif,size):
    out = tomopy.misc.corr.remove_outlier1d(data,dif=dif,size=size,axis=1)
    return out

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
        #load in A-hutch metal filter information 
        try:
            filt1 = str(data_hdf['measurement']['instrument']['filters']['Filter_1_Material'][...])
        except KeyError:
            filt1 = []
        try:
            filt2 = str(data_hdf['measurement']['instrument']['filters']['Filter_2_Material'][...])
        except KeyError:
            filt2 = []
        #load in magnification
        try:
            mag = data_hdf['measurement']['instrument']['detector']['magnification'][...]
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

    #Handle beam hardening
    if beam_hardening:
        #import the filters that were used in data acquisition    
        bh_dir = dir(bh)
        filters = {}
        filters = {bh.Be_filter:750.0}
        if filt1 != 'Open':
            mat1 = filt1.split('_')[0]
            thick1 = float(filt1.split('_')[-1][:-2])
            temp1 = [j for j in range(len(bh_dir)) if mat1+'_filter' in bh_dir[j]][0]
            filters[getattr(bh,bh_dir[temp1])]=thick1

        if filt2 != 'Open':
            mat2 = filt2.split('_')[0]
            thick2 = float(filt2.split('_')[-1][:-2]) 
            temp2 = [j for j in range(len(bh_dir)) if mat2+'_filter' in bh_dir[j]][0]
            filters[getattr(bh,bh_dir[temp2])]=thick2

        filtered_spectrum = bh.fapply_filters(filters)
        #plt.plot(bh.spectral_energies,filtered_spectrum)
        #plt.show()
        bh.sample_material = bh.Fe_filter
        global coeffs
        coeffs = bh.fcompute_lookup_coeffs(bh.spectral_energies,filtered_spectrum,sample_material,scint_material,scintillator_thickness) / (pixel_size * 1e4)
        print("\nCoefficients for fit for transmission to pathlength conversion complete.")
       
    #set slices to reconstruct 
    row_slice = slice(start_row, end_row+1, row_step)
    column_slice = slice(start_col, end_col+1, 1)
    
    #Import the projection images
    projection_data_full = fread_projection_images(data_filename,row_slice,column_slice)

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
        dsize = float((projection_data_full.nbytes)/1e9)
    print("Size of dataset to be reconstructed is %0.2f GB" %dsize) 
          
#    plt.figure(1)
#    plt.imshow(projection_data_full[:,0,:],cmap='gray') #plot sinogram
#    plt.title('After Normalization')
#    plt.colorbar()
#    plt.show()
       
    #print("Center of rotation is at %s" %rotation_center) 
    
    #start reconstruction of dataset chunks
    ta = time.time()
    # Loop chunks
    for smin in np.arange(0,projection_data_full.shape[1],chunk-2*margin):
        tc=time.time()
        smax = smin + chunk
        if smax>projection_data_full.shape[1]: 
            smax=projection_data_full.shape[1]
        print(('\nWorking on slices {0:d} to {1:d} of image ({2:d} slices total)'.format(smin + start_row,smax + start_row,projection_data_full.shape[1])))
 
        #Extract chunk for further processing
        projection_data_subset = projection_data_full[:,smin:smax,:] # Extract chunk
        #Stripe removal
        projection_data_subset = fremove_stripes(projection_data_subset)
        subd_size = float((projection_data_subset.nbytes)/1e9)
        print("Size of chunked data is %0.2f GB" %subd_size)
       
        #Correct for beam hardening, if desired.  Now in terms of pathlength
        if beam_hardening:
            projection_data_subset = fapply_beam_hardening_projection(projection_data_subset)
        
        #If desired, perform phase retrieval
        if phase_retrieval:
            projection_data_subset = fapply_phase_retrieval(projection_data_subset)
        
        print(('\tTime elapsed prepping chunk {0:6.3f} s.'.format(time.time() - tc)))
        
        #Take care of datasets where we do a 0-360 degree rotation to increase FOV
        working_data_subset, working_theta, working_rotation_center = freregister_data(projection_data_subset,theta,rotation_center,rotation_center)
        del(projection_data_subset)
        print(("Rotation center = {:0.2f} pixels.".format(working_rotation_center)))
        newdata = float((working_data_subset.nbytes)/1e9)
        print("Size of chunked data is %0.2f GB" %newdata)

        # Mask huge values, NaNs and INFs from the filtered data
        maxv=1e5
        working_data_subset[np.isnan(working_data_subset)]=0
        working_data_subset[np.isinf(working_data_subset)]=np.nanmax(working_data_subset)*np.sign(working_data_subset[np.isinf(working_data_subset)])
        working_data_subset[np.abs(working_data_subset)>maxv]=maxv*np.sign(working_data_subset[np.abs(working_data_subset)>maxv])
        old_width = working_data_subset.shape[2]
         
        # Pad the array
        (working_data_subset, new_center) = fpad_image(working_data_subset,working_rotation_center,pad_fraction_width)
        pad_size = float((working_data_subset.nbytes)/1e9)
        print("Padded array is %0.2f GB" %pad_size)
        
        # Tomographic reconstruction
        data_recon = freconstruct_projections(working_data_subset,working_theta,new_center,old_width)
        recon_size = float((data_recon.nbytes)/1e9)
        print("Recon array is %0.2f GB" %recon_size)

       
        # Ring removal
        t0=time.time()
        print("\nRemoving rings...")
        data_recon = tomopy.remove_ring(data_recon.copy(),ncore=ncore)
        #mask outer edges with circular mask
        data_recon = tomopy.circ_mask(data_recon, axis=0, ratio=0.95)
        print('\tElapsed %i sec\n' % (time.time()-t0))

        # Write tiff stack out
        if output_path_tiffstack is not None:
            fwrite_tiff_stack(data_recon,smin,projection_data_full.shape[1])
        
        #Clear memory to keep from blowing up the memory usage.    
        del(working_data_subset, data_recon)
        
#    plt.figure()
#    plt.imshow(projection_data_full[20,...])
#    plt.title('After Beam Hardening')
#    plt.colorbar()
#    plt.show()
         
### MAIN #######################################################################################################################

if __name__ == "__main__":
    start_time = time.time()
    #find file name and center of rotation from text file
    with open(centers_file,'r') as f:
        modlines = []
        for line in f:
            modlines.append(line)
    if data_key == 'ALL':
        dat = copy.deepcopy(modlines)
    else:
        dat = [j for j in modlines if data_key in j]
    for cond in dat:
        if cond.startswith("#"):
            pass
        else:
            data_filename = cond.split(' ')[0]
            rotation_center = float(cond.split(' ')[1][:-2])       
            #create location for saving data
            file_path  = os.path.dirname(data_filename)
            basename = os.path.basename(os.path.normpath(file_path))
            output_path_tiffstack = file_path+"/ReconTest2_"+basename+"/" # None for no TIFF output
            print("Working on %s" %basename)
            print("Images will be written to %s"%output_path_tiffstack)
                
            #Give slice objects with the image ROI
            p,r,c = fget_dataset_shape(data_filename,row_slice= slice(None,None,1),column_slice=slice(None,None,1))
            # Fraction of slices to load into RAM at once (smaller number reduces I/O overhead)
            chunk = 100 # Number of slices to reconstruct at once (RAM-limited)
            margin = 1  # Num slices at ends to not write (due to phase retrieval boundary effects)
            rowmin = 0
            rowmax = 20
            if rowmin is None:
                rowmin = 0
            if rowmax is None:
                rowmax = r
            rchunk=10
            row_step = 1 # Phase retrieval requires step of 1 
            for rows in np.arange(rowmin,rowmax,rchunk):
                start_row = rows
                end_row = rows+rchunk-1
                if end_row>rowmax: 
                    end_row=rowmax
                print(start_row,end_row)
                if start_col is None:
                   start_col = 0
                if end_col is None:
                   end_col = c 
                ftomopy_script()
            print('Total elapsed time = %.1f min' % ((time.time() - start_time)/60.))

      

    
