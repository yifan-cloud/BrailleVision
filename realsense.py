''' Source: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
'''

# fundamental package for scientific computing
import numpy as np
# 2D plotting library producing publication quality figures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Intel RealSense cross-platform open-source API
import pyrealsense2 as rs
print("Environment Ready")

''' Getting the data
'''



#---- GATHER RAW ----#

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("./bags/stairs.bag")
profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()
  
# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()
print("Frames Captured")


colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = [8, 4]
plt.imshow(colorized_depth)
plt.title("Raw Depth")
plt.show()


#---- GATHER FILTERED ----#


depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()

hole_filling = rs.hole_filling_filter()

profile = pipe.start(cfg)
frames = []
frames = []
for x in range(10):
    frameset = pipe.wait_for_frames()
    frames.append(frameset.get_depth_frame())

pipe.stop()
print("Frames Captured")

for x in range(10):
    frame = frames[x]
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
plt.title("Filtered")
plt.imshow(colorized_depth)
plt.show()

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

#---- DOWNSAMPLE FILTERED ----#

downsampled_rgb = bin_ndarray(colorized_depth, (4, 4, 3), 'mean').astype('int')
plt.imshow(downsampled_rgb)
plt.title('Downsampled')
plt.show()

