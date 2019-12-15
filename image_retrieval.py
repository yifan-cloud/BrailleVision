"""
Contains functions to call to get depth/RGB images from the realsense camera
"""

# fundamental package for scientific computing
import numpy as np
# Intel RealSense cross-platform open-source API
import pyrealsense2 as rs

#----- initializing code -----
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("./bags/stairs.bag")
profile = pipe.start(cfg)

onDepthMode = False

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()
pipe.stop()

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# init smoothing filters
decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()

hole_filling = rs.hole_filling_filter()
#-----------------------------

# must call startDepthMode every time we start on/switch to depth-vibration mode
def startDepthMode():
    profile = pipe.start(cfg)
    onDepthMode = True

# returns the current depth frame as a numpy array
# must call startDepthMode every time we start on/switch to depth-vibration mode
def getBinnedDepthArray():
    if not onDepthMode:
        return None

    frameset = pipe.wait_for_frames()
    depth_frame = frameset.get_depth_frame()

    # filter frame for smoothness
    depth_frame = filterDepthImg(depth_frame)

    # depth_frame class -> numpy array
    depth_image = np.asanyarray(depth_frame.get_data())

    binned_depth_arr = bin_ndarray(depth_image, (4, 4), 'mean')#.astype('int')

    return binned_depth_arr

def filterDepthImg(frame):
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame) # only useful w/ continuous stream
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

    return frame

# must call endDepthMode every time we leave depth-vibration mode
def endDepthMode():
    pipe.stop()
    onDepthMode = False

# returns the current color frame as a numpy array
def getColorImg():
    profile = pipe.start(cfg)
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    pipe.stop()

    # frame(?) class -> numpy array
    color_image = np.asanyarray(color_frame.get_data())

    return color_image

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    SOURCE: https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray

def main():
    startDepthMode()
    print( getBinnedDepthArray() )
    endDepthMode()

if __name__ == "__main__":
    main()