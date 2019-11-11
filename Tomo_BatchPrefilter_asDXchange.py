#!/usr/bin/env python

"""
Tomography image stack preprocessor service. Processes HDF5 files from tomography fly scans,
performs median filtering on repeated scans to remove zingers, and averages the frames to 
improve SNR. If there is only one repetition of data, zinger removal using TomoPy will be performed.
If there are two repetitions, zingers will be replaced with the minimum value at that pixel location. 
The code also calculates required chunksize assuming that the user wants to keep 1/3 of the
system's available RAM free. This is set by the variable "threshold", and can be changed  

This file belongs in the top-level directory as it takes its location as the current working directory.
The script walks through all of the folders inside this top-level directory, searches for HDF5 files, and checks
if a prefiltered file already exists. If it doesn't, then it will create one.

Alan Kastengren, XSD, APS
Katie Matusik, XSD, APS


Edits by Aniket Tekawade, April 16, 2019:

It was found that the multiprocessing Queue approach would randomly skip writing projections to the HDF5 file.
Defining two separate queues for processing and writing is leading to occassional loss in data.
Also, multiprocessing was disabled for tomopy.outlier (nproc = 1) since overall processs is being parallelized.
Replaced with multiprocessing Pool, which I have found more robust. Also, sequentialized processing and writing into one function "fprocess_images"
Did not compare processing time with the previous code, however this solves the issue.

Edits by Aniket Tekawade, April 18, 2019:

Edits by Alan Kastengren, started August 2, 2019
Copy the first HDF5 file and replace the arrays in /exchange to make the prefiltered file.
Make a new group /replicate_meta to hold everything but the /exchange group.
Simplify the calculations to just use medians when we have more than two replicates.
"""

import time
import os
import shutil
import sys
from multiprocessing import cpu_count, Pool
import glob
import h5py
import numpy as np
import tomopy
import psutil
import functools

oneGB = 1024**3
nbytes_16bit = 2
RAM_limit = 0.2 # While loading data to RAM, use this fraction as threshold so the rest of the RAM is available for other processes.
nprocs = 30 # Use nprocs = cpu_count() as default.


'''Function definitions''' 

# This function process a set of images at the same projection angle.
# E.g. if 5 repeatable datasets were obtained, this function process 5 images at the same projection angle.

def Parallelize(ListIn, f, procs = -1, **kwargs):
    
    # This function packages the "starmap" function in multiprocessing for Python 3.3+ to allow multiple iterable inputs for the parallelized function.
    # ListIn: any python list such that each item in the list is a tuple of non-keyworded arguments passable to the function 'f' to be parallelized.
    # f: function to be parallelized. Signature must not contain any other non-keyworded arguments other than those passed as iterables.
    # Example:
    # def multiply(x, y, factor = 1.0):
    #   return factor*x*y
    # X = np.linspace(0,1,1000)
    # Y = np.linspace(1,2,1000)
    # XY = [ (x, Y[i]) for i, x in enumerate(X)] # List of tuples
    # Z = IP.Parallelize_MultiIn(XY, multiply, factor = 3.0, procs = 8)
    # Create as many positional arguments as required, but remember all must be packed into a list of tuples.
    
    if type(ListIn[0]) != tuple:
        ListIn = [(ListIn[i],) for i in range(len(ListIn))]
    
    reduced_argfunc = functools.partial(f, **kwargs)
    
    if procs == -1:
        procs = cpu_count()
        

    if procs == 1:
        OutList = [reduced_argfunc(*ListIn[iS]) for iS in range(len(ListIn))]
    else:
        p = Pool(processes = procs)
        OutList = p.starmap(reduced_argfunc, ListIn)
        p.close()
        p.join()
    
    return OutList


def fprocess_images(image_data):    
    '''Code to perform zinger removal on one projection.
    '''
    #if there is only one image set, do zinger removal with TomoPy
    if image_data.shape[0]==1:
        out = tomopy.remove_outlier(image_data,zinger_level,ncore = 1) # DO NOT CHANGE ncore here! ...
        # This function is kept serial so that the several instances of this function can be parallelized. Trying to change ncore here will result in child processes which are not supported in multiprocessing.
        out = out[0]
    #if there are two datasets, replace zingers by min value at each pixel location
    elif image_data.shape[0]==2:
        out = np.mean(image_data, axis=0, dtype=np.float32).astype(np.uint16)
        minval = np.fmin(image_data[0, ...],image_data[1, ...])
        mask = (out - minval) > int(zinger_level / 2)
        out[mask] = minval[mask]
    #otherwise, just use the median value for the pixel
    elif image_data.shape[0]>1:
        out = np.median(image_data,axis=0).astype(np.uint16)
    return out


def fstructure_new_file(hdf_filenames,output_filename):
    '''Make a new HDF5 file in the same format as the old files.
    Copy the first file.
    Put all groups not in /exchange into a new /replicate_meta group.
    '''
    with h5py.File(output_filename,'w') as output_hdf:
        with h5py.File(hdf_filenames[0] ,'r+') as first_input:
            first_input['/defaults'].copy(first_input['/defaults'], output_hdf)
            first_input['/measurement'].copy(first_input['/measurement'], output_hdf)
            first_input['/process'].copy(first_input['/process'], output_hdf)
            output_hdf.create_group('exchange')
            for key in first_input['/exchange'].attrs.keys():
                output_hdf['/exchange'].attrs[key] = first_input['/exchange'].attrs[key]
        print("First replicate meta data copied as the output file to maintain structure.")
        output_hdf.create_group('replicate_meta')
        for i,fname in enumerate(hdf_filenames):
            with h5py.File(fname,'r') as input_hdf:
                group_name = 'replicate_{0:d}'.format(i)
                output_hdf['replicate_meta'].create_group(group_name)
                for sub_group in input_hdf.keys():
                    if sub_group == 'exchange':
                        continue
                    else:
                        input_hdf.copy(input_hdf[sub_group],output_hdf['replicate_meta'][group_name])

def fprocess_hdf_stack(hdf_filenames,output_filename,chunksize):
    '''Process a set of HDF files.
    Make an output file structured in DXchange format.
    Process for zinger removal.
    '''
    #Make the output file.
    fstructure_new_file(hdf_filenames, output_filename)
    #Compute the filtered bright and dark fields
    file_list = []
    bright_list = []
    dark_list = []
    print("Loading in bright and dark field data...")
    for fname in hdf_filenames:
        file_list.append(h5py.File(fname,'r'))
        #save all of the bright and dark fields as separate lists
        bright_list.extend(file_list[-1]['exchange']['data_white'][...])
        dark_list.extend(file_list[-1]['exchange']['data_dark'][...])
    bright_list = np.asarray(bright_list)
    dark_list = np.asarray(dark_list)
    #run the bright and dark images through median filtering to get a single image out
    bright_field = fprocess_images(bright_list)
    dark_field = fprocess_images(dark_list)
    #Write the bright and dark data to the output file.
    with h5py.File(output_filename,'r+') as output_hdf:
        output_hdf['/exchange'].create_dataset('data_white', data=bright_field)
        output_hdf['/exchange'].create_dataset('data_dark', data=dark_field)
        for key in file_list[0]['/exchange/data_white'].attrs.keys():
            output_hdf['/exchange/data_white'].attrs[key] = file_list[0]['/exchange/data_white'].attrs[key]
        for key in file_list[0]['/exchange/data_dark'].attrs.keys():
            output_hdf['/exchange/data_dark'].attrs[key] = file_list[0]['/exchange/data_dark'].attrs[key]
    print("Done writing bright and dark fields to HDF5 file.")

    #process chunks of data so that we don't run out of memory
    data_shape =  h5py.File(hdf_filenames[0],'r')['exchange']['data'].shape
    num_proj = data_shape[0]
    ints = range(num_proj)
    
    #Make the dataset in the output HDF file
    with h5py.File(output_filename, 'r+') as output_hdf:
        output_hdf['/exchange'].create_dataset('data', shape=data_shape, dtype=np.uint16)
        for key in file_list[0]['/exchange/data'].attrs.keys():
            output_hdf['/exchange/data'].attrs[key] = file_list[0]['/exchange/data'].attrs[key]
        
    
    print("Loading %i images into memory..."%chunksize)
    
    for i in range(0,num_proj,chunksize):
        #Preallocate an array to hold the data
        chunk = ints[i:i+chunksize]
        total_dataset_shape = file_list[0]['/exchange/data'].shape
        if i + chunksize >= total_dataset_shape[0]:
            chunk = range(i, total_dataset_shape[0])
        print([len(file_list), len(chunk), total_dataset_shape[1], total_dataset_shape[2]])
        data_array = np.empty([len(chunk), len(file_list), total_dataset_shape[1], total_dataset_shape[2]], dtype=np.uint16)
        print('Starting to read in data for this chunk.')
        time0 = time.time()
        for i, files in enumerate(file_list):
            data_array[:,i,...] = files['/exchange/data/'][chunk,...]
        #data_array = np.swapaxes(data_array, 0, 1)
        data_list = [data_array[iL, ...] for iL in range(data_array.shape[0])]
        print("Elapsed time to load images %i-%i is %0.2f minutes." %(chunk[0],chunk[-1],(time.time() - time0)/60))
        
        data_list = np.asarray(Parallelize(data_list, fprocess_images, procs = nprocs))
        
        with h5py.File(output_filename,'a') as fout:
            fout['/exchange/data'][chunk,...] = data_list
    return


if __name__ == '__main__':  
    start_time = time.time()
    main_folder = os.getcwd()
    zinger_level=3000
    all_folders = next(os.walk('.'))[1]
    
    
    
    #all_folders = glob.glob(main_folder+'/LGPS_5_OCP_000')
    for item in all_folders:
        if item == 'tomoPyScriptUtilities':continue
        raw_images_filenames = glob.glob(main_folder+'/'+item + '/*.hdf5')
        

        #raw_images_filenames = glob.glob(item+'/*.hdf5')
        #check for prefiltered hdf5 file in folder
#        sys.exit()
        if raw_images_filenames:
            here = [j for j in raw_images_filenames if 'Prefiltered.hdf5' in j]
            if here:
                print("Prefiltering has already been completed. Prefiltered images are saved in %s."%os.path.basename(here[0]))
                pass
            else:
                output_dir = item+'/'
                if output_dir[-1] != '/':
                    output_dir=output_dir + '/'   
                if raw_images_filenames:
                    print('Raw HDF5 files are in directory %s' %item)
                    if len(raw_images_filenames)>1:
                        print('There are %i raw image files. These will be averaged together.' %(len(raw_images_filenames)))
                    elif len(raw_images_filenames)==1:
                        print('There is %i raw image file.' %(len(raw_images_filenames)))
                    else:
                        print('There are %i raw image files.' %(len(raw_images_filenames)))
                    tempname = os.path.basename(raw_images_filenames[0]).split('.')[0]
#                    tempname_split = tempname.split('_')[:-1]
                    tempname_split = tempname.split('_')[:] #THIS HAS BEEN CHANGED FOR THE CELL TOMO - BAS 4/3/2019
                    output_filename = output_dir+'_'.join(tempname_split) + '_Prefiltered.hdf5'
                    print("Data will be written to %s"%output_filename)
                    #get size of images assuming 16 bit to figure out chunksize
                    
                    
                    
                    with h5py.File(raw_images_filenames[0],'r') as f:
                        #m = projections, n = rows, o = columns
                        (m,n,o) = f['exchange']['data'].shape
                        size = (nbytes_16bit*m*n*o)/(oneGB) 
                        totsize = size*(len(raw_images_filenames)+1) # +1 accounts for one prefiltered set
                        mem = psutil.virtual_memory()
                        avmem = mem.available/(oneGB) #system memory that is available and not swap
                        #assume that we want 1/3 of the system memory to be free for other things and can therefore use 2/3 of it
                        threshold = avmem*RAM_limit 
                        if totsize>threshold:
                            #back out how many images can be processed with threshold memory
                            chunksize = int((threshold*(oneGB))/(len(raw_images_filenames)+1)/(nbytes_16bit*n*o))
                        else:
                            chunksize = int(m)

                    #sys.exit()
    
                    fprocess_hdf_stack(raw_images_filenames,output_filename,chunksize)
                else:
                    print('Raw image files could not be found. Please check file location and names.')
                print("Elapsed time is %0.2f minutes" %((time.time() - start_time)/60))
        else:
            print("No 'hdf5' files found in %s. Skipping directory." %item)
            pass
 
