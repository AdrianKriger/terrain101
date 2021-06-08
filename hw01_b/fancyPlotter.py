# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:25:22 2021

@author: arkriger
"""

#from laspy.file import File
import pyvista as pv
import xarray as xr
import numpy as np

#import matplotlib.pyplot as plt
#from matplotlib import cm

def read_raster(filename):
    """
    Helpful: http://xarray.pydata.org/en/stable/auto_gallery/plot_rasterio.html
    """
    # Read in the data
    data = xr.open_rasterio(filename)
    values = np.asarray(data)
    nans = values == -9999
    if np.any(nans):
        # values = np.ma.masked_where(nans, values)
        values[nans] = np.nan
    # Make a mesh
    xx, yy = np.meshgrid(data['x'], data['y'])
    zz = values.reshape(xx.shape) # will make z-comp the values in the file
    # zz = np.zeros_like(xx) # or this will make it flat
    mesh = pv.StructuredGrid(xx, yy, zz)
    mesh['data'] = values.ravel(order='F')
    return mesh

# ~ change these dtm/dsm
topo1 = read_raster('./idwPDAL/CPUT_Lidar_dtm_idwPDAL.tif')
topo2 = read_raster('./idwQUAD/CPUT_Lidar_dtm_idwQUAD.tif')
topo3 = read_raster('./tinLaplace/CPUT_Lidar_dtm_tinLaplace.tif')
topo4 = read_raster('./tinCnst/CPUT_Lidar_dtm_tinCnst.tif')

p = pv.Plotter(window_size=[750, 450], notebook=False, shape=(2, 2))#, off_screen=True)
pv.set_plot_theme("document")
p.subplot(0, 0)
p.add_text("idwPDAL", font_size=8)
p.add_mesh(topo1, cmap='terrain', show_scalar_bar=False) #cmap='jet' ~ for dsm
p.subplot(0, 1)
p.add_text("idwQUAD", font_size=8)
p.add_mesh(topo2, cmap='terrain', show_scalar_bar=False) #cmap='jet' ~ for dsm
p.subplot(1, 0)
p.add_text("tinLaplace", font_size=8)
p.add_mesh(topo3, cmap='terrain', show_scalar_bar=False) #cmap='jet' ~ for dsm
p.subplot(1, 1)
scalar_bar_args={'title': ''}
p.add_text("tinCnst", font_size=8)
p.add_mesh(topo4, cmap='terrain', show_scalar_bar=True, scalar_bar_args=scalar_bar_args) #cmap='jet' ~ for dsm

p.set_background('white')
p.link_views()  # link all the views
p.show()
