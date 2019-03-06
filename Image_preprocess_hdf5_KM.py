#!/usr/bin/env python
"""
White beam tomography image stack preprocessor service

Reads sets of images as they are saved, removes zingers, mean-filters and saves the filtered images
at 16 bits for post-reconstruction. This can save a lot of time not having to do the ensemble filtering
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
import h5py
import matplotlib.pyplot as plt
from skimage import img_as_int
# Start General settings #######################################################
zinger_level=3000
raw_images_dir = os.getcwd() + '/'
print(raw_images_dir)
raw_images_filenames = glob.glob(raw_images_dir + "*_Tomo__*.hdf5")
bright_filename = glob.glob(raw_images_dir + "*_Bright__*.hdf5")[0]
dark_filename = glob.glob(raw_images_dir + "*_Dark__*.hdf5")[0]
#print(bright_filename, dark_filename)
output_dir = raw_images_dir#+"_prefiltered"
# End General settings #######################################################

# Take a list of images and perform zinger removal, then mean filter
def fproc_image_subset(image_data,zinger_level=3000):
    '''Performs prefiltering on a stack of images.
    Input arguments:
    image_data: 3D numpy array where dimenions are [replication,image_num,rows,cols]
    zinger_level:
    Output:
    2D numpy array in the form [rows,cols] for the prefiltered image
    '''
    # Find the median for each pixel of the prefiltered image
    med = np.median(image_data,axis=0)
    # Loop through the image set
    for j in range(image_data.shape[0]):
        replicate = image_data[j,...]
        mask = replicate - med > zinger_level
        replicate[mask] = med[mask] # Substitute with median
        image_data[j,...] = replicate # Put data back in place
        #print("Number of zingers detected = " + str(np.sum(mask)))
    #del med
    
    # Mean filter the stack, now the zingers are gone.
    # Keep as int16 to save space  
    return np.mean(image_data,axis=0,dtype=np.float32).astype(np.uint16)

def fprocess_hdf_stack(hdf_filenames,output_filename=''):
    '''Processes repeated scans in HDF5 format, writing a single HDF5 file.
    Input arguments:
    hdf_filenames: names of input HDF5 files
    output_filename: name of the file to be written
    directory (optional): name of parent directory
    '''
    hdf_files = []
    data_list = []
    #Open up the HDF files
    for fname in hdf_filenames:
        print(fname)
        hdf_files.append(h5py.File(fname,'r'))
        data_list.append(hdf_files[-1]['entry']['data']['data'])
    working_array = np.empty((len(hdf_files),data_list[0].shape[1],data_list[0].shape[2]))
    print(working_array.shape)
    #Open up an output HDF5 file
    if not output_filename:
        output_filename = hdf_filenames[0].split('Tomo_')[0] + 'Prefiltered.hdf5'
    with h5py.File(output_filename,'w') as output_hdf:
        output_hdf.create_dataset('Prefiltered_images',data_list[0].shape,dtype=np.uint16)
        #Loop through the projection angles
        for i in range(data_list[0].shape[0]):
            if i % 10 == 0:
                print(i)
            for j in range(len(hdf_files)):
                working_array[j,...] = data_list[j][i,...]
            output_hdf['Prefiltered_images'][i,...] = fproc_image_subset(working_array)
    #Close the input hdf5 files
    for i in hdf_files:
        i.close()
    return

def fprocess_hdf_bright_dark(hdf_filenames,output_filename=''):
    '''Processes an HDF5 file that contains a stack of images to be prefiltered.
    Useful for bright/dark files.
    '''
    output_filename = ''
    if not output_filename:
        output_filename = hdf_filenames.split('.')[0][:-3] + 'Prefiltered.hdf5'
    with h5py.File(output_filename,'w') as output_hdf:
        #Open the input HDF5 file
        with h5py.File(hdf_filenames,'r') as input_hdf:
            data = input_hdf['entry']['data']['data'][...]
            output_hdf.create_dataset('Prefiltered_images',data[0].shape,dtype=np.uint16)
            output_hdf['Prefiltered_images'][...] = fproc_image_subset(data)
            #s=plt.imshow(fproc_image_subset(data),cmap='gray')
            #plt.colorbar(s)
    return

if __name__ == '__main__':
    raw_images_dir = os.getcwd() + '/'
    #print(raw_images_dir)
    raw_images_filenames = glob.glob(raw_images_dir + "*_Tomo_*.hdf5")
    print(raw_images_filenames)
    fprocess_hdf_stack(raw_images_filenames)
    fprocess_hdf_bright_dark(bright_filename)
    fprocess_hdf_bright_dark(dark_filename)

    
