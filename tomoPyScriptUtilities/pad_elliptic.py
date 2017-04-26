# Elliptical padding function for tomography of fuel spray nozzles
# Daniel Duke
# X-ray Fuel Spray Group
# Argonne National Laboratory

# Started Jan 8, 2016
# Last Updated Apr 27, 2016

import numpy as np

def pad_elliptic(data,radius=None,center=None,pad_amt=None):
    prev_width=data.shape[2]

    # edge value pad by 50% of width on each side
    if pad_amt is None: pad_amt=prev_width/2

    # First do edge value pad
    print "\tPadding by edge value first..."
    data = np.pad(data, ((0,0),(0,0),(pad_amt,pad_amt)),mode='edge') 

    # by default, if elliptical radius is not specified asssume that it reaches zero by the edge of the window 
    if radius is None: 
        radius = data.shape[2]/2 - 1
        print "\tUsing half-width of image as radius (%.1f px)" % radius
    elif radius<0: # if the radius is negative, interpret as a scaling factor:try using the maximum magnitude * -radius /2
        radius = -radius*np.nanmax(data)/2.
        print "\tEstimating ellipse radius from inline magnitude as %.1f px" % radius
    else:
        print "\tRadius = %.1f px" % radius
    
    # by default, if center of rotation is not specified assumed to be in the middle.
    if center is None: center = data.shape[2]/2.
    else: center = np.float(center+pad_amt)
    
    # Make ellptical profiles for left & right sides
    r = np.abs(np.arange(data.shape[2])-center)
    
    print "\tComputing ellipses..."
    ellipses_L = data[...,0:pad_amt] * np.nan_to_num(np.sqrt(1 - (r[0:pad_amt]/radius)**2))
    ellipses_R = data[...,-pad_amt:] * np.nan_to_num(np.sqrt(1 - (r[-pad_amt:]/radius)**2))
    
    data[...,0:pad_amt]=np.nan_to_num(ellipses_L)
    data[...,-pad_amt:]=np.nan_to_num(ellipses_R)
    
    '''
    # Debugging code
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i in range(data.shape[0]):
        ax.plot(data[i,:,:].T,marker='o')
    plt.show()
    '''
    return data
