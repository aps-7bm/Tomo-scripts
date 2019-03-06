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
"""

import time
import os
from multiprocessing import Process, Queue, JoinableQueue, cpu_count
import glob
import h5py
import numpy as np
import tomopy
import psutil

'''Function definitions''' 

# The consumer function takes data off of the Queue
def consumer(inqueue,output):    
    # Run indefinitely
    while True:         
        # If the queue is empty, queue.get() will block until the queue has data
        all_data = inqueue.get()
        if all_data:
            #n is the index corresponding to the projection location 
            n, image_data = all_data
            #if there is only one image set, do zinger removal with TomoPy
            if image_data.shape[0]==1:
                out = tomopy.remove_outlier(image_data,zinger_level,ncore = cpu_count()-1)
            #if there are two datasets, replace zingers by min value at each pixel location
            elif image_data.shape[0]==2:
                # If the difference in pixel value between two frames is too high,
                #  assume the higher value is a zinger, and hence invalid
                minval = np.fmin(image_data[0,...],image_data[1,...])
                data_range = np.abs(image_data[0,...],image_data[1,...])
                mask = data_range > zinger_level
                image_data[:,mask] = minval[mask]
                out = np.mean(image_data,axis=0,dtype=np.float32).astype(np.uint16)
            #otherwise, replace zingers with median and average stack 
            elif image_data.shape[0]>2:
                #Find the median for each pixel of the prefiltered image
                med = np.median(image_data,axis=0)
                #Loop through the image set
                for j in range(image_data.shape[0]):
                    replicate = image_data[j,...]
                    mask = replicate - med > zinger_level
                    replicate[mask] = med[mask] # Substitute with median
                    image_data[j,...] = replicate # Put data back in place
                out = np.mean(image_data,axis=0,dtype=np.float32).astype(np.uint16)
            output.put((n,out))
        else:
            break

#function for writing out HDF5 file        
def write_hdf(output,output_filename):
    #create output HDF5 file
    while True:
        args = output.get()
        if args:
            i,data = args
            with h5py.File(output_filename,'a') as fout:
                fout['Prefiltered_images'][i,...] = data
        else:
            break

def fprocess_hdf_stack(hdf_filenames,output_filename,chunksize):
    #take care of bright and dark fields first
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
    bright_field = fprocess_hdf_bright_dark(bright_list)
    dark_field = fprocess_hdf_bright_dark(dark_list)

    #put everything in output HDF5 file
    fout = h5py.File(output_filename,'a') 
    fout.create_dataset('Bright_field',data = bright_field,dtype=np.uint16)
    fout.create_dataset('Dark_field',data = dark_field,dtype=np.uint16)
    print("Done writing bright and dark fields to HDF5 file.")

    #copy things over from one of the data HDF5 files
    with h5py.File(fname,'r') as fin:
        fin.copy('measurement',fout)
        #fix the filter names and other attributes that should be strings
        filt1 = fout['measurement']['instrument']['filters']['Filter_1_Material'][...].ravel()
        filt2 = fout['measurement']['instrument']['filters']['a givenFilter_2_Material'][...].ravel()
        del fout['measurement']['instrument/filters/Filter_1_Material']
        del fout['measurement']['instrument/filters/Filter_2_Material']
        fout['measurement']['instrument']['filters']['Filter_1_Material'] = ''.join(chr(i) for i in [d for d in filt1 if d])
        fout['measurement']['instrument']['filters']['Filter_2_Material'] = ''.join(chr(i) for i in [d for d in filt2 if d]) 
        strings = ['firmware_version','manufacturer','model','software_version']
        for keys in strings:
            temp = fout['measurement']['instrument']['detector'][keys][...].ravel()
            del fout['measurement']['instrument']['detector'][keys]
            fout['measurement']['instrument']['detector'][keys] = ''.join(chr(i) for i in [d for d in temp if d])
        if 'table' in fout['measurement'].keys():
            del fout['measurement']['table']
    print("Done copying attributes over to HDF5 file.")

    #process chunks of data so that we don't run out of memory
    totsize = h5py.File(hdf_filenames[0],'r')['exchange']['data'].shape[0]
    data_shape =  h5py.File(hdf_filenames[0],'r')['exchange']['data'].shape
    fout.create_dataset('Prefiltered_images',data_shape,dtype=np.uint16)
    fout.close()
    ints = range(totsize)
    chunkSize= chunksize

    #initialize how many consumers we would like working 
    num_consumers = cpu_count()*2

    #Create the Queue objects
    inqueue = JoinableQueue()
    output = Queue()     
        
    #start process for writing HDF5 file
    proc = Process(target=write_hdf, args=(output,output_filename))
    proc.start()
       
    print("Loading %i images into memory..."%chunkSize)
    for i in range(0,totsize,chunkSize):
        time0 = time.time()
        chunk = ints[i:i+chunkSize]
        data_list = []
        #Make a list of the HDF5 datasets we are reading in
        for files in file_list:
            #shape is (angles, rows, columns)
            data_list.append(files['exchange']['data'][chunk,...])
        data_list = np.asarray(data_list)
        print("Elapsed time to load images %i-%i is %0.2f minutes." %(chunk[0],chunk[-1],(time.time() - time0)/60))
     
        consumers = []       
        
        #Create consumer processes
        for i in range(num_consumers):
            p = Process(target=consumer, args=(inqueue,output))
            consumers.append(p)
            p.start()
        
        for n in range(data_list.shape[1]):
            #Feed data into the queue
            inqueue.put((chunk[n],data_list[:,n,...]))
        
        #Kill all of the processes when everything is finished    
        for i in range(num_consumers):
            inqueue.put(None)
         
        for c in consumers:
            c.join()
        print("Elapsed time to process images %i-%i is %0.2f minutes." %(chunk[0],chunk[-1],(time.time() - time0)/60))
        
    time.sleep(1)
    output.put(None)
    proc.join()     

    #Close the input HDF5 files.
    for hdf_file in file_list:
        hdf_file.close()
    print("Input HDF5 files closed.") 
    return

def fprocess_hdf_bright_dark(data):
    '''Processes an HDF5 file that contains a stack of images to be prefiltered.
    Useful for bright/dark files.
    '''
    med = np.median(data,axis=0)
    for j in range(data.shape[0]):
            replicate = data[j,...]
            mask = replicate - med > zinger_level
            replicate[mask] = med[mask] # Substitute with median
            data[j,...] = replicate # Put data back in place
    # Mean filter the stack now that the zingers are gone.
    # Keep as int16 to save space
    img = np.mean(data,axis=0,dtype=np.float32).astype(np.uint16)
    return img

if __name__ == '__main__':  
    start_time = time.time()
    main_folder = os.getcwd()
    zinger_level=3000
    all_folders = next(os.walk('.'))[1]
    #all_folders = glob.glob(main_folder+'/22_5E_6_BC_CC*')
    for item in all_folders:
        raw_images_filenames = glob.glob(main_folder+'/'+item + '/*.hdf5')
        #raw_images_filenames = glob.glob(item+'/*.hdf5')
        #check for prefiltered hdf5 file in folder
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
                    tempname_split = tempname.split('_')[:-1]
                    output_filename = output_dir+'_'.join(tempname_split) + '_Prefiltered.hdf5'
                    print("Data will be written to %s"%output_filename)
                    #get size of images assuming 16 bit to figure out chunksize
                    with h5py.File(raw_images_filenames[0],'r') as f:
                        #m = projections, n = rows, o = columns
                        (m,n,o) = f['exchange']['data'].shape
                        size = (2*m*n*o)/(1e9)
                        totsize = size*(len(raw_images_filenames)+1) # +1 accounts for one prefiltered set
                        mem = psutil.virtual_memory()
                        avmem = mem.available/(1e9) #system memory that is available and not swap
                        #assume that we want 1/3 of the system memory to be free for other things and can therefore use 2/3 of it
                        threshold = avmem*(2/3.) 
                        if totsize>threshold:
                            #back out how many images can be processed with threshold memory
                            chunksize = int((threshold*1e9)/(len(raw_images_filenames)+1)/(2*n*o))
                        else:
                            chunksize = int(m)
                    fprocess_hdf_stack(raw_images_filenames,output_filename,chunksize)
                else:
                    print('Raw image files could not be found. Please check file location and names.')
                print("Elapsed time is %0.2f minutes" %((time.time() - start_time)/60))
        else:
            print("No 'hdf5' files found in %s. Skipping directory." %item)
            pass
 
