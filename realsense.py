''' Source: https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/depth_filters.ipynb
'''

import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
print("Environment Ready")

''' Getting the data
'''

# Setup:
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


''' Visualizing the data
'''

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

plt.rcParams["axes.grid"] = False
plt.rcParams['figure.figsize'] = [8, 4]
plt.imshow(colorized_depth)
plt.show()

''' Applying Filters 
'''

# Decimation - reduces spatial resolution and adds hole filling

decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)
decimated_depth = decimation.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

# Spatial

spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)
filtered_depth = spatial.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

# Temporal
profile = pipe.start(cfg)

frames = []
for x in range(10):
    frameset = pipe.wait_for_frames()
    frames.append(frameset.get_depth_frame())

pipe.stop()
print("Frames Captured")

temporal = rs.temporal_filter()
for x in range(10):
    temp_filtered = temporal.process(frames[x])
colorized_depth = np.asanyarray(colorizer.colorize(temp_filtered).get_data())
plt.imshow(colorized_depth)
plt.show()

# Hole Filling

hole_filling = rs.hole_filling_filter()
filled_depth = hole_filling.process(depth_frame)
colorized_depth = np.asanyarray(colorizer.colorize(filled_depth).get_data())
plt.imshow(colorized_depth)
plt.show()

# Putting Everything Together
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

for x in range(10):
    frame = frames[x]
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    frame = hole_filling.process(frame)

colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
plt.imshow(colorized_depth)
plt.show()


