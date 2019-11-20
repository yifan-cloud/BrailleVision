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

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file("./bags/stairs.bag")
profile = pipe.start(cfg)

frames = []
for x in range(10):
    frameset = pipe.wait_for_frames()
    frames.append(frameset.get_depth_frame())

pipe.stop()
print("Frames Captured")

colorizer = rs.colorizer()

depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

decimation = rs.decimation_filter()
decimation.set_option(rs.option.filter_magnitude, 4)

spatial = rs.spatial_filter()
spatial.set_option(rs.option.holes_fill, 3)

temporal = rs.temporal_filter()

hole_filling = rs.hole_filling_filter()



for x in range(10):
    frame = frames[x]
    frame = decimation.process(frame)
    frame = depth_to_disparity.process(frame)
    frame = spatial.process(frame)
    # frame = temporal.process(frame)
    frame = disparity_to_depth.process(frame)
    # frame = hole_filling.process(frame)

colorized_depth = np.asanyarray(colorizer.colorize(frame).get_data())
plt.imshow(colorized_depth)
plt.show()

depth_frame = frame


# downsample
# bin colorized depth into 16 equal partitions (later -- user configure width and depth)
fig = plt.figure()
ax = fig.gca(projection='3d')
xs, ys, zs = list(), list(), list()
# plot_frames = list()
darr = np.asanyarray(depth_frame.get_data())
x_bin = darr.shape[0] // 4
y_bin = darr.shape[1] // 4
y_max = darr.shape[1]
for x_idx in range(4):
    x_frame = darr[x_bin*x_idx : x_bin*(x_idx+1)]
    for y_idx in range(4):
        frame = x_frame[:, y_bin*y_idx : y_bin*(y_idx+1)]
        avg_depth = np.mean(frame)
        print(x_idx, y_idx, avg_depth)
        # plot_frames.append((x_idx, y_idx, avg_depth)) # (row, col, avg_depth)
        xs.append(x_idx)
        ys.append(y_idx)
        zs.append(avg_depth)

ax.scatter(xs, ys, zs)
plt.show()

'''
# app filters

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
'''