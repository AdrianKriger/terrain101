# -*- coding: utf-8 -*-
# env/pyvista02
"""
started; on Sat Jun  5 20:25:22 2021

@author: arkriger 

Basic script to read an elevation geotif (DTM) and contour.shapefile 
transform to PyVista objects, plot and interogate.
"""

import itertools
import pyvista as pv
import rasterio
import numpy as np
import geopandas as gpd

def read_raster(filename):
    """
    read elevation geotif; return mesh
    """
    # Read in the data
    ds = rasterio.open('tasmania.tif')
    dem = np.flipud(ds.read(1))
    nans = dem == ds.nodatavals
    if np.any(nans):
        # values = np.ma.masked_where(nans, values)
        dem[nans] = np.nan

    x = np.arange(ds.bounds[0], ds.bounds[2], ds.res[0])
    y = np.arange(ds.bounds[1], ds.bounds[3], ds.res[1])
    
    # Creating meshgrid
    x, y = np.meshgrid(x, y)
    
    # Creating Structured grid
    grid = pv.StructuredGrid(x, y, dem)
    # Assigning elevation values to grid
    grid["Elevation"] = dem.ravel(order="F")
    
    return grid

def contours_to_3d(gdf):
    """create vtk object from contour.shp for plotting with PyVista
    """
    #create emtpy dict to store the partial unstructure grids
    lineTubes = {}
    
    #iterate over the points
    for index, values in gdf.iterrows():
        cellSec = []
        linePointSec = []
    
        #iterate over the geometry coords
        zipObject = zip(values.geometry.xy[0],values.geometry.xy[1], 
                        itertools.repeat(values.elev))
        for linePoint in zipObject:
            linePointSec.append([linePoint[0],linePoint[1],linePoint[2]])
    
        #get the number of vertex from the line and create the cell sequence
        nPoints = len(list(gdf.loc[index].geometry.coords))
        cellSec = [nPoints] + [i for i in range(nPoints)]
    
        #convert list to numpy arrays
        cellSecArray = np.array(cellSec)
        cellTypeArray = np.array([4])
        linePointArray = np.array(linePointSec)
    
        partialLineUgrid = pv.UnstructuredGrid(cellSecArray,cellTypeArray,linePointArray)   
        #we can add some values to the point
        partialLineUgrid.cell_arrays["elev"] = values.elev
        lineTubes[str(index)] = partialLineUgrid
    
    #merge all tubes and export resulting vtk
    lineBlocks = pv.MultiBlock(lineTubes)
    lineGrid = lineBlocks.combine()

    return lineGrid

#contours
cntDf = gpd.read_file('contours.shp')
vtk_cnt = contours_to_3d(cntDf)
#raster
dem = read_raster('tasmania.tif')

p = pv.Plotter(window_size=[750, 450], notebook=False)
pv.set_plot_theme("document")
scalar_bar_args={'title': ''}
p.add_mesh(dem, cmap='terrain', 
           show_scalar_bar=True, scalar_bar_args=scalar_bar_args)

# make the dem pickable ~ return coord and height on click
dargs = dict(name='labels', font_size=14)
def callback(mesh, pid):
    point = dem.points[pid]
    label = ['xy: {}\nelevation: {}'.format([round(dem.points[::,0][pid], 2), 
                                             round(dem.points[::,1][pid], 2)], 
                                     dem["Elevation"][pid])]
    p.add_point_labels(point, label, bold=False,
                       italic=True, text_color='w', 
                       shape_opacity = 0.3, **dargs)

p.enable_point_picking(callback=callback, show_message=True, 
                       color='red', point_size=7, 
                       use_mesh=True, show_point=True)

## display the contours aswell
# p.add_mesh(vtk_cnt, color='white', 
#             show_scalar_bar=False, line_width=1, opacity=0.3)

p.set_background('white')
#p.enable_fly_to_right_click(callback=None)
p.show(cpos="xy")#screenshot='terrain.png')
