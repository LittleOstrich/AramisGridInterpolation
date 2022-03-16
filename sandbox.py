import numpy as np
from data import dataLoader
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib

# Written by Florian Thamm

# Settings
path = r"C:\Users\geiss\OneDrive\Desktop\LainProject\Python\AramisGridInterpolation\data\cleanSamples\dreieck_raster2.csv"
h = 100  # width of target resolution
w = 100  # height of target resolution
margin = [0.7, 0.7, 0.7, 0.7]  # percentage of surface what is being kept x left, x right, y left, y right
# margin = 4*[1]
# Prepare data
df = dataLoader.loadDataByPath(path)
coords = df.values[:, 1:4]
vectors = df.values[:, 4:7]

# Compute grid
xx, yy = np.mgrid[np.min(coords[:, 0]) * margin[0]: np.max(coords[:, 0]) * margin[1]: complex(0, h),
         np.min(coords[:, 1]) * margin[2]: np.max(coords[:, 1] * margin[3]): complex(0, w)]

# interpolation of z value
zz = griddata(coords[:, 0:2], coords[:, 2], (xx, yy), method="cubic")
# zz = coords[:, 2]
# interpolation of displacement
vx = griddata(coords[:, 0:2], vectors[:, 0], (xx, yy), method="linear")
vy = griddata(coords[:, 0:2], vectors[:, 1], (xx, yy), method="linear")
vz = griddata(coords[:, 0:2], vectors[:, 2], (xx, yy), method="linear")

v_grid = np.linalg.norm(np.concatenate([vx[None], vy[None], vz[None]]), axis=0)
v_points = np.linalg.norm(vectors, axis=1)

# # __________________ Plotting ________________________
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
wireframe_stride = h
surf = ax1.plot_wireframe(xx, yy, zz, alpha=1, rstride=int(h / wireframe_stride), cstride=int(w / wireframe_stride))
# surf = ax1.plot_surface(xx, yy, zz, alpha = 1)
scatter = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c="red", marker=".", alpha=0.3)
ax1.set_title("Fit through surface")

# color stuff (color as 4th dimension)
percentiles = [0, 90]
min, max = np.percentile(v_grid, percentiles[0]), np.percentile(v_grid, percentiles[1])
v_grid = np.clip(v_grid, min, max)

# surface
ax2.set_title("Fit color encoded with displacement lengths")
norm = matplotlib.colors.Normalize(min, max)
m = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
m.set_array([])
fcolors = m.to_rgba(v_grid)
surf = ax2.plot_surface(xx, yy, zz, facecolors=fcolors, alpha=0.7)

# scatter
# v_points =  np.clip(v_points, min, max)
# scatter = ax2.scatter(coords[:, 0], coords[:, 1], v_points, c = v_points, cmap="jet", alpha = 1)
plt.show()
