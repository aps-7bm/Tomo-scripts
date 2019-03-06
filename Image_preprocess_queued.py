#!/usr/bin/env python
"""
White beam tomography image stack preprocessor service.
Processes HDF5 files from white beam fly scans, performs median filtering on
repeated scans to remove zingers, and averages the frames to improve S/N.

Alan Kastengren, XSD, APS
"""

from multiprocessing import Queue

import sys, copy, glob, os, gc, time, os.path
import numpy as np
import h5py
# Start General settings #######################################################
zinger_level=3000
raw_images_dir = os.getcwd() + '/'
print(raw_images_dir)
raw_images_filenames = glob.glob(raw_images_dir + "*_Tomo__*.hdf5")
bright_filename = glob.glob(raw_images_dir + "*_Bright__*.hdf5")[0]
dark_filename = glob.glob(raw_images_dir + "*_Dark__*.hdf5")[0]
output_dir = raw_images_dir
num_processors = 6
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
    
    # Mean filter the stack, now that the zingers are gone.
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



    
