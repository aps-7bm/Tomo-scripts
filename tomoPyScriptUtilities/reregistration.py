#!/usr/bin/env python
'''
Image re-registration for tomography scans with projections spanning more than 180 degrees.
Function freregister will widen the sinogram and flip the >180 degree projections back onto the 0-180 degree projections to create a virtual sinogram with a 0 to 180 degree spin.

At present, the code throws away the part of the image to the right hand side of the center of rotation, rather than averaging between the two views. This is done because wobbling of the rotation stage could blur this overlapped region.

Daniel Duke (dduke@anl.gov)

Modifications
04-05-2017: ALK modifications to make code easier to read.  
            Also better account for the fact that the center of rotation may be a float.  
            Don't fill new array with zeros, so we can save some time.

Started: Apr 29, 2016
'''

import numpy as np
import time

# Data is array of [projections, slices, width].
# Theta is array of radian values for each projection.
# Center is pixel location in width for center of rotation.
def freregister(data, theta, center):
    t0=time.time()
    print "\nRe-registering projections for span > 180 degrees..."

    # Find starting point for angles > 180 degrees
    #Account for roundoff error
    delta_theta = (theta[-1] - theta[0]) / (float(theta.shape[0]) - 1.0)
    index_180 = np.argmax(theta >= (np.pi - delta_theta / 10.0)) #Returns first argument where this is True
    # Make new theta array
    new_theta = theta[:index_180]
    #Verify that we have the same number of angles after 180 degrees
    print(np.allclose(new_theta+np.pi,theta[index_180:2*index_180]))
    #Round the center of rotation to the nearest half pixel.
    print(center)
    rotation_center = np.round(center * 2.0) / 2.0
    
    # Find new width of array
    old_image_center = (data.shape[2] - 1.0) / 2.0      #In between middle two pixels
    overlap_width = data.shape[2]  - np.abs(old_image_center - rotation_center)
    new_width = data.shape[2]  + int(2.0 * np.abs(old_image_center - rotation_center))
    new_center = float(new_width - 1) / 2.0
    print("Overlap between views 180 degrees apart = {0:0.1f} pixels.".format(overlap_width))
    print("New image width = {0:0.1f} pixels.".format(new_width))
    print("New image center = {0:0.1f} pixels.".format(new_center))

    # Make new data array 
    print "\tExisting data shape=",data.shape
    new_shape=(index_180,data.shape[1],new_width)
    print "\tRe-registering projections into a new array of shape = ",new_shape
    data_rereg = np.empty(new_shape,dtype=data.dtype) # Save the time of filling with zeros vs. np.zeros!

    # Figure out mapping of old data to the new array
    #Determine if the rotation axis is to the left or to the right of the center of the image
    if rotation_center >= old_image_center:
        old_image_slice = slice(0,int(np.ceil(rotation_center-0.01)),1)
        new_image_slice_0_180 = slice(0,int(np.ceil(new_center-0.01)),1)
        new_image_slice_180_360 = slice(data_rereg.shape[2]-1,int(np.floor(center + 0.01)),-1)
    else:
        old_image_slice = slice(int(np.ceil(rotation_center-0.01)),data.shape[2],1)
        new_image_slice_0_180 = slice(int(np.ceil(new_center-0.01)),data_rereg.shape[2],1)
        new_image_slice_180_360 = slice(int(np.floor(new_center+0.01)),None,-1)
    
    data_rereg[...,new_image_slice_0_180]=data[:index_180,:,old_image_slice]

        
    data_rereg[...,new_image_slice_180_360]=data[index_180:2*index_180,:,old_image_slice]
    
    print "\tOut: mean=%.2f, min=%.2f, max=%.2f, std=%.2f" % (np.mean(data),data.min(),data.max(),np.std(data))
    print 'Total elapsed time = %.1f min' % ((time.time() - t0)/60.)

    '''
    # Show sinograms! For debugging
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(121)
    c_=ax.imshow(data[:,data.shape[1]/2,:])
    plt.colorbar(c_)
    plt.title("Input sinogram")
    ax=fig.add_subplot(122)
    c_=ax.imshow(data_rereg[:,data_rereg.shape[1]/2,:])
    plt.colorbar(c_)
    plt.title("Reregistered sinogram")
    plt.suptitle("Sinogram samples")
    plt.show()
    '''
    
    return data_rereg, new_theta, new_center
