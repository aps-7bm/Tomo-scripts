#!/usr/bin/env python
"""
White beam tomography image stack preprocessor service

Reads sets of images as they are saved, removes zingers, mean-filters and saves the filtered images
at 32 bits for post-reconstruction. This can save a lot of time not having to do the ensemble filtering
during the reconstruction step. ROI cropping is also possible at this step.

Daniel Duke
X-ray Fuel Spray
Energy Systems Division
Argonne National Laboratory

Started: Oct 28, 2015

Edits: Oct 29, 2015 
       Alan Kastengren changed the image filter to use a median difference outlier detection instead of an
       absolute threshold.

       Feb 25, 2016
       Dan Duke added a timeout to quit the program if no new images found after several minutes, and avoid
       double-underscore in filename by stripping trailing underscore from image name prefix.

       Feb 26, 2016
       Added friendly warning if no images found (misspelt prefix etc)
	   
	   July 12, 2016
	   Automate the program so that it reads the directory and prefix from the camera soft IOC.
	   Ideally, this will eliminate the need for manual edits of the source code.

       July 14, 2016
       Changed wildcard on line 151 to search specifically for images with _nnnnn.tif at the end.
"""

import sys, copy, glob, os, gc, time, os.path
import numpy as np
from tifffile import *
import epics

# General settings #######################################################

crop_x=None
crop_y=None
zinger_level=3000
directory_PV = epics.PV('7bmPG1:TIFF1:FilePath_RBV')
image_prefix_PV = epics.PV('7bmPG1:TIFF1:FileName_RBV')
image_repeats_PV = epics.PV('7bmPG1:cam1:NumImages_RBV')

raw_images_dir = directory_PV.get(as_string=True)
raw_images_prefix = image_prefix_PV.get(as_string=True)
n_repeat = int(image_repeats_PV.get())
print raw_images_dir
print raw_images_prefix
print n_repeat

#Handle case where we are running this on Linux
if raw_images_dir[:2] == 'Z:':
    raw_images_dir = '/home/beams/7BMB/SprayData/' + raw_images_dir[3:].replace('\\','/')
print raw_images_dir
output_dir = raw_images_dir#+"_prefiltered"
output_images_prefix = raw_images_prefix+"prefiltered_"

nprocesses=2
quit_timeout=600 # sec
########################################################################



# Attempt to load joblib
try:
    from joblib import Parallel, delayed
except ImportError:
    print "Sorry - multithreading not available :-("
    nprocesses=1


# Take a list of images and perform zinger removal, then mean filter
def fproc_image_subset(all_images,output_filename,crop_x=None,crop_y=None,zinger_level=3000):

    # Load images into a list (this way, they can be any size)
    # Load first image to get the numpy array size
    I = imread(all_images[0])

    if crop_y is not None: I=I[crop_y[0]:crop_y[1],:]
    if crop_x is not None: I=I[:,crop_x[0]:crop_x[1]]
    #print I.shape
    data_subset = np.zeros((len(all_images),I.shape[0],I.shape[1]))
    data_subset[0,:,:] = I
    for i,filename in enumerate(all_images[1:]):
        I = imread(filename)
        if crop_y is not None: I=I[crop_y[0]:crop_y[1],:]
        if crop_x is not None: I=I[:,crop_x[0]:crop_x[1]]
        data_subset[i+1,:,:] = I
    #data_subset = np.array(data_subset) # Convert list of arrays to 3D ndarray
    
    # Remove zingers before filtering - replace them with median value
    med = np.median(data_subset,axis=0)
    zinger_counter = 0
    zinger_level_intensity = zinger_level
    # Loop through the image set
    for j in range(len(all_images)):
        slc=data_subset[j,...]
        #zinger_counter = np.sum(slc-med > zinger_level_intensity)
        #print zinger_counter
        #slc[slc>zinger_level_intensity] = med[slc>zinger_level_intensity] # Substitute with median
        slc[slc-med > zinger_level_intensity] = med[slc-med > zinger_level_intensity] # Substitute with median
        data_subset[j,...]=slc # Put data back in place
        #del slc
    del med
    
    # Mean filter the stack, now the zingers are gone.
    # Keep as float32 for precision
    #flat = np.mean(data_subset,axis=0,dtype=np.float32).astype(np.float32)
    
    
    # Write 32bit float TIFF
    imsave(output_filename,np.mean(data_subset,axis=0,dtype=np.float32).astype(np.float32))
    del data_subset
    gc.collect()
    
    return


# Function to check if a processed image exists, and generate it otherwise
# This function is a wrapper for multithreading, and is called by fconvert_all_images
def _fworker_thread(i,n_repeat,output_images_prefix,output_dir,indices,raw_images_list):
    output_filename = output_images_prefix+'%05i.tif' % (i/n_repeat)
        
    # If output file not already made ...
    if not os.path.exists(output_dir+'/'+output_filename):
        # Check for missing raw images in the group
        any_missing = np.any([not k in indices for k in np.arange(i,i+n_repeat)])
        if any_missing:
            print 'Could not process images %i-%i, one or more were missing. Skipping.' % (i,i+n_repeat-1)
            return
            
        # Go ahead and process the set of images
        glob_indices = [indices.index(k) for k in np.arange(i,i+n_repeat)]
        raw_image_set = [ raw_images_list[k] for k in glob_indices ]
        fproc_image_subset(raw_image_set,output_dir+'/'+output_filename,crop_x,crop_y,zinger_level)
        print 'Done: images %i-%i -> %s' % (i,i+n_repeat-1,output_filename)
        sys.stdout.flush()
    
    return


# Main function to call to run the preprocessor.
# Scan through a directory of images and filter all the ones that haven't been made yet.
def fconvert_all_images(raw_images_dir,raw_images_prefix,output_dir,output_images_prefix,\
                        n_repeat,crop_x,crop_y,zinger_level):
    
    # Read raw image directory listing
    if raw_images_dir[-1]=='_': raw_images_dir=raw_images_dir[:-1]
    raw_images_list = glob.glob(raw_images_dir+'/'+raw_images_prefix+'_?????.tif')
    print '\n',len(raw_images_list),'raw images found in directory'

    if len(raw_images_list)==0:
        print 'No images found! Check file and directory names.'
        time.sleep(10)
        exit()
    
    # Get the numbers out of the filenames
    indices = [ int(os.path.splitext(os.path.basename(fn))[0].split('_')[-1]) for fn in raw_images_list ]
    print 'Starting from image',np.min(indices),'preprocessing every',n_repeat,'images'
    
    # Make output directory
    if not os.path.exists(output_dir):
        print 'Creating',output_dir
        os.makedirs(output_dir)

    # Check for which files are not already done
    indices_todo=[]
    print 'Checking for images already processed...'
    for i in np.arange(np.min(indices),np.max(indices),n_repeat):
        output_filename = output_images_prefix+'%05i.tif' % (i/n_repeat)
        if not os.path.exists(output_dir+'/'+output_filename):
            indices_todo.append(i)

    # If there is nothing to do, just return
    if len(indices_todo)<1:
        print 'Nothing to do.'
        return 0
    
    print len(indices_todo),'images to create'
    
    # Sweep indices range, creating output image if missing
    if nprocesses <= 1:
        # Serial iterator
        for i in indices_todo:
            _fworker_thread(i,n_repeat,output_images_prefix,output_dir,indices,raw_images_list)
    else:
        # Parallel iterator using joblib
        Parallel(n_jobs=nprocesses,verbose=0)(delayed(_fworker_thread)(i,n_repeat,\
                        output_images_prefix,output_dir,indices,raw_images_list)\
                        for i in indices_todo)
    return 1




if __name__ == "__main__":
    print "Tomography image ensemble preprocessor"
    print "D. Duke - X-ray Fuel Spray - Argonne National Laboratory"
    print "(dduke@anl.gov)"
    print "-----------------------------------------------------------------------------"
    
    # Loop forever ...
    t=time.time()
    while True:
        status = fconvert_all_images(raw_images_dir,raw_images_prefix,output_dir,output_images_prefix,\
                                     n_repeat,crop_x,crop_y,zinger_level)
        
        print "-----------------------------------------------------------------------------"
        print "Finished processing directory. Looping back to check for more missing ones..."
        print "-----------------------------------------------------------------------------"
        time.sleep(5)
        if status > 0: t=time.time() # reset timeout clock
        if (time.time()-t)> quit_timeout: break
    
    exit()
