#!/usr/bin/env python
"""
White beam tomography image stack preprocessor service.
Processes HDF5 files from white beam fly scans, performs median filtering on
repeated scans to remove zingers, and averages the frames to improve S/N.

Alan Kastengren, XSD, APS
"""

import multiprocessing

import glob, time, os
import numpy as np
import h5py

# Start General settings #######################################################
zinger_level=3000
raw_images_dir = os.getcwd() + '/'
raw_images_filenames = glob.glob(raw_images_dir + "*_Tomo__*.hdf5")
bright_filename = glob.glob(raw_images_dir + "*_Bright__*.hdf5")[0]
dark_filename = glob.glob(raw_images_dir + "*_Dark__*.hdf5")[0]
output_dir = raw_images_dir

# End General settings #######################################################

#Set up multiprocessing
class Angle_Consumer(multiprocessing.Process):
    '''Process to consume projection data.
    '''
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print 'Exiting'
                self.task_queue.task_done()
                break
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return
    
class Angle_Task():
    '''Class giving task to perform zinger removal and averaging.
    Inputs
    projection: int giving the projection number
    image_data: 3D numpy array of dimensions (replicate,rows,columns)
    '''
    def __init__(self, projection, image_data):
        self.projection = projection
        self.image_data = image_data
    def __call__(self):
        # Find the median for each pixel of the prefiltered image
        med = np.median(self.image_data,axis=0)
        # Loop through the image set
        for j in range(self.image_data.shape[0]):
            replicate = self.image_data[j,...]
            mask = replicate - med > zinger_level
            replicate[mask] = med[mask] # Substitute with median
            self.image_data[j,...] = replicate # Put data back in place
        
        # Mean filter the stack, now that the zingers are gone.
        # Keep as int16 to save space
        #Place into the JoinableQueue for the writing
        return Writer_Task(self.projection,
                           np.mean(self.image_data,axis=0,dtype=np.float32).astype(np.uint16))
#         return (self.projection,
#                            np.mean(self.image_data,axis=0,dtype=np.float32).astype(np.uint16))
 
class Writer_Consumer(multiprocessing.Process):
    '''Process to write projection data to an HDF file.
    '''
    def __init__(self, task_queue,hdf_filename,data_shape):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.hdf_filename = hdf_filename
        with h5py.File(hdf_filename,'w') as hdf_file:
            hdf_file.create_dataset('Prefiltered_images',data_shape,dtype=np.uint16)
        #self.dataset = self.hdf_file['Prefiltered_images']
 
    def run(self):
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('Exiting writer.')
                self.task_queue.task_done()
                break
            next_task(self.hdf_filename)
            self.task_queue.task_done()
        return
 
class Writer_Task():
    '''Class giving task to perform zinger removal and averaging.
    '''
    def __init__(self, projection, image_data):
        self.projection = projection
        self.image_data = image_data
        
    def __call__(self,hdf_filename):
        #Write data to the hdf_dataset
        with h5py.File(hdf_filename,'r+') as hdf_file:
            hdf_file['Prefiltered_images'][self.projection,...] = self.image_data
        return
     

def fprocess_hdf_stack(hdf_filenames,output_filename=''):
    '''Processes repeated scans in HDF5 format, writing a single HDF5 file.
    Structured so we don't have to read all of the data in before executing.
    Input arguments:
    hdf_filenames: names of input HDF5 files
    output_filename: name of the file to be written
    directory (optional): name of parent directory
    '''
    data_list = []
    file_list = []
    #Make a list of the HDF5 datasets we need
    for fname in hdf_filenames:
        print(fname)
        file_list.append(h5py.File(fname,'r'))
        #Make an array of datasets to be used for reading data
        data_list.append(file_list[-1]['entry']['data']['data'])
    
    #Open up an output HDF5 file
    if not output_filename:
        output_filename = hdf_filenames[0].split('Tomo_')[0] + 'Prefiltered.hdf5'
    
    #Make multiprocessing Queues to handle the processing
    # Establish communication queues
    raw_queue = multiprocessing.JoinableQueue()
    prefiltered_queue = multiprocessing.JoinableQueue()
    
    # Start consumers
    num_angle_consumers = multiprocessing.cpu_count() - 2
    print 'Creating %d consumers' % num_angle_consumers
    angle_consumers = [Angle_Consumer(raw_queue, prefiltered_queue)
                  for i in range(num_angle_consumers)]
    for w in angle_consumers:
        w.start()
    writer_consumer = Writer_Consumer(prefiltered_queue,output_filename,data_list[0].shape)
    writer_consumer.start()
    
    #Loop through the projection angles
    for i in range(data_list[0].shape[0]):
        working_array = np.empty((len(hdf_filenames),data_list[0].shape[1],data_list[0].shape[2]))
        if i % 10 == 0:
            print("Starting on projection {0:d} of {1:d}".format(i,data_list[0].shape[0]))
            print("Queuing: {0:d} ready for zinger removal, {1:d} ready to write.".format(
                                        raw_queue.qsize(),prefiltered_queue.qsize()))
        for j in range(len(hdf_filenames)):
            working_array[j,...] = data_list[j][i,...]
        raw_queue.put(Angle_Task(i,working_array))
    print("Finished reading in data files.")
    print("{0:d} projections left to process.".format(raw_queue.qsize()))
    print("{0:d} projections left to write.".format(prefiltered_queue.qsize()))
    
    #Close the input HDF5 files.
    for hdf_file in file_list:
        hdf_file.close()
    print("Input HDF5 files closed.")
    
    #Add in poison pills
    for w in range(num_angle_consumers):
        raw_queue.put(None)
    #Make sure to finish the processing before putting in writer poison pills.
    raw_queue.join()
    print("Finished processing.")
    print("{0:d} projections left to write.".format(prefiltered_queue.qsize()))
    time.sleep(1)
    #Wait for the writing to complete.
    prefiltered_queue.put(None)
    prefiltered_queue.join()
    print("Finished writing.")
    
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
        processor = Angle_Task(0,data)
        writer_task = processor()
        
        output_hdf.create_dataset('Prefiltered_images',data[0].shape,dtype=np.uint16)
        output_hdf['Prefiltered_images'][...] = writer_task.image_data
    return

if __name__ == '__main__':
    fprocess_hdf_stack(raw_images_filenames)
    fprocess_hdf_bright_dark(bright_filename)
    fprocess_hdf_bright_dark(dark_filename)