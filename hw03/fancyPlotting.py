# -*- coding: utf-8 -*-
"""
Created on Sun May 16 14:57:57 2021

@author: arkriger
"""
#import sys
from laspy.file import File
import pyvista as pv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm


## ~~ first one

rFile = File('./ahn3/rural_classified.las', mode = "r")
r_points = np.array((rFile.x, rFile.y, rFile.z, rFile.Classification)).transpose()
r_df = pd.DataFrame(r_points, columns=['X', 'Y', 'Z', 'Classification'])
r_points = r_df[r_df['Classification'] == 2]
r_points = np.vstack((r_points['X'], r_points['Y'], r_points['Z'])).T

r_cloud = pv.PolyData(r_points)

p = pv.Plotter(window_size=[450, 250], notebook=False, off_screen=True)
pv.set_plot_theme("document")

r_cloud["elevation"] =  r_points[:,2] # g['HeightAboveGround']  #  
cmap = plt.cm.get_cmap("jet")

def zoom(plotter, value):
    if not plotter.camera_set:
        plotter.camera_position = plotter.get_default_cam_pos()
        plotter.reset_camera()
        plotter.camera_set = True
    plotter.camera.Zoom(value)
    plotter.render()
    
# Monkey patch it
pv.Plotter.zoom = zoom

p.add_mesh(r_cloud, point_size=2, render_points_as_spheres=True, 
            scalar_bar_args={'title': 'rural:ground'}, cmap=cmap)# stitle="Elevation")
                #, cmap=plt.cm.get_cmap("viridis", 5))
p.set_background("white")
p.zoom(1.5)
p.show(auto_close=False)
path = p.generate_orbital_path(n_points=85, shift=r_cloud.length)
p.open_gif('r_orbit.gif')
p.orbit_on_path(path, write_frames=True)
p.close()

# then the other
# =============================================================================
# 
# uFile = File('./ahn3/urban_classified.las', mode = "r")
# u_points = np.array((uFile.x, uFile.y, uFile.z, uFile.Classification)).transpose()
# u_df = pd.DataFrame(u_points, columns=['X', 'Y', 'Z', 'Classification'])
# u_points = u_df[(u_df['Classification'] == 2) & (u_df['Classification'] != 7) & (u_df['Z'] <= 0.7)]
# u_points = np.vstack((u_points['X'], u_points['Y'], u_points['Z'])).T
# 
# p = pv.Plotter(window_size=[450, 250], notebook=False, off_screen=True)
# pv.set_plot_theme("document")
# 
# u_cloud = pv.PolyData(u_points)
# u_cloud["elevation"] =  u_points[:,2] # g['HeightAboveGround']  #  
# cmap = plt.cm.get_cmap("jet")
# 
# p.add_mesh(u_cloud, point_size=2, render_points_as_spheres=True, 
#             scalar_bar_args={'title': 'urban:ground'}, cmap=cmap)
# p.set_background("white")

# p.add_mesh(r_cloud, point_size=2, render_points_as_spheres=True, 
#             scalar_bar_args={'title': 'rural:ground'}, cmap=cmap)# stitle="Elevation")
#                 #, cmap=plt.cm.get_cmap("viridis", 5))
# p.set_background("white")
# p.zoom(1.5)
# 
# p.show(auto_close=False)
# path = p.generate_orbital_path(n_points=85, shift=u_cloud.length)
# p.open_gif('u_orbit.gif')
# p.orbit_on_path(path, write_frames=True)
# p.close()
# =============================================================================
