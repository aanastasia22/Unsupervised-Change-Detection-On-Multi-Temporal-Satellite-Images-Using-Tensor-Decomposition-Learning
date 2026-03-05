import numpy as np
import rasterio
import os
import scipy.io as sio
from collections import namedtuple

Info = namedtuple('Info', 'start height')

# returns height, width, and position of the top left corner of the largest
#  rectangle with the given value in mat
def max_size(mat, value=1):
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_size_start, start_row = max_rectangle_size(hist), 0
    for i, row in enumerate(it):
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        mss = max_rectangle_size(hist)
        if area(mss) > area(max_size_start):
            max_size_start, start_row = mss, i+2-mss[0]
    return max_size_start[:2], (start_row, max_size_start[2])

# returns height, width, and start column of the largest rectangle that
#  fits entirely under the histogram
def max_rectangle_size(histogram):
    stack = []
    top = lambda: stack[-1]
    max_size_start = (0, 0, 0) # height, width, start of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                max_size_start = max(
                    max_size_start,
                    (top().height, pos - top().start, top().start),
                    key=area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        max_size_start = max(max_size_start, (height, pos - start, start),
            key=area)

    return max_size_start

def area(size): return size[0]*size[1]


def form_time_series(path,bands):
    # Read the images of a location of an event and form the time series and the corresponding mask of true changes
    # Input: 
    #       path        : Path to the folder of a location of an event (string)
    #       bands       : Highest resolution spectral bands (array of size num_bands x 1)
    # Output:
    #       Dictionary consists of:
    #         'data'    : Image time series (array of size Height x Width x num_bands x num_frames)
    #         'changes' : Mask of true changes (array of size Height x Width)

    path_img = os.path.join(path,'S2')
    path_labels = os.path.join(path,'changes')

    sequences = os.listdir(path_img)
    num_frames = len(sequences) # Number of frames in the time series
    num_bands = len(bands)

    # Find the coordinates of non-nan values in each image of the time series
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(num_frames):
        img = rasterio.open(os.path.join(path_img,sequences[i]))
        img = img.read()
        bi = ~np.isnan(img[0,:,:])
        cord = max_size(bi,True)
        x1.append(cord[1][0])
        y1.append(cord[1][1])
        x2.append(x1[i]+cord[0][0])
        y2.append(y1[i]+cord[0][1])
    # Define the new coordinates to crop the images
    x1_new = max(x1)
    x2_new = min(x2)
    y1_new = max(y1)
    y2_new = min(y2)

    # Form the time series of size Height x Width x num_bands x num_frames
    for i in range(len(sequences)):
        img = rasterio.open(os.path.join(path_img,sequences[i]))
        img = img.read()
        img = np.transpose(img,(1,2,0))
        img = img[:,:,bands]
        img = img[x1_new:x2_new,y1_new:y2_new,:]

        if i == 0:
            data = np.zeros((img.shape[0],img.shape[1],num_bands,num_frames))
        data[:,:,:,i] = img    
    # Take the corresponding mask of true changes
    changes = np.squeeze(rasterio.open(os.path.join(path_labels,os.listdir(path_labels)[0])).read())
    changes = changes[x1_new:x2_new,y1_new:y2_new]
    
    return {'data':data,'changes':changes}