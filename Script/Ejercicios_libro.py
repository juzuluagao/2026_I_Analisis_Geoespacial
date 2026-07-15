import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from pointpats import centrography
import seaborn as sbn
import contextily as ctx

gdf = gpd.read_file("https://github.com/edieraristizabal/ModeloMultinivel/raw/refs/heads/main/DATA/df_catchments_spatial.gpkg")

gdf['x'] = gdf.geometry.centroid.x
gdf['y'] = gdf.geometry.centroid.y

mean_center = centrography.mean_center(gdf[['x', 'y']])
med_center = centrography.euclidean_median(gdf[['x', 'y']])

gdf.crs

os.environ['PROJ_LIB'] = r'c:\Analisis_Geoespacial\2026_I_Analisis_Geoespacial\.venv\Lib\site-packages\pyproj\proj_dir\share\proj'

joint_axes = sbn.jointplot(x='x', y='y', data=gdf, s=0.75, height=9)

xmin, xmax = joint_axes.ax_joint.get_xlim()
ymin, ymax = joint_axes.ax_joint.get_ylim()

joint_axes.ax_joint.scatter(*mean_center, color='red', marker='x', s=50, label='Mean Center')
joint_axes.ax_marg_x.axvline(mean_center[0], color='red')
joint_axes.ax_marg_y.axhline(mean_center[1], color='red')

joint_axes.ax_joint.scatter(*med_center, color='limegreen', marker='o', s=50, label='Median Center')
joint_axes.ax_marg_x.axvline(med_center[0], color='limegreen')
joint_axes.ax_marg_y.axhline(med_center[1], color='limegreen')
joint_axes.ax_joint.legend()

joint_axes.ax_joint.set_xlim(xmin, xmax)
joint_axes.ax_joint.set_ylim(ymin, ymax)
plt.show()