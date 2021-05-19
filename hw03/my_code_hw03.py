#-- mycode_hw03.py
#-- GEO1015.2019--hw03
#-- [## ~~ arkriger] 
#-- [YOUR STUDENT NUMBER]
import os
import subprocess
import math

import numpy as np
import json
import pdal 

from osgeo import gdal, osr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pyvista as pv

# triangulation for ground filtering algorithm and TIN interpolation 
#import startin

# kdtree for IDW interpolation
from scipy.spatial import cKDTree
from invdisttree import Invdisttree
import scipy.spatial
from scipy.interpolate import LinearNDInterpolator
#from scipy.interpolate import NearestNDInterpolator


def write_ascii(zg, origin, resolution, jparams, out_asc):
    """
    write esri ascii grid given an array + origin and resolution of a .las dataset
    """
    
    zg[np.isnan(zg)] = -9999
    
    xgmin = origin[0]
    ygmin = origin[1]
    ngrows = resolution[1]
    ngcols = resolution[0]
       
    zg = np.flipud(zg)
     
    # ASCII file header
    header = "ncols %d \nnrows %d\nxllcorner %f\nyllcorner %f\ncellsize %f\nnodata_value -9999" \
            % (int(ngcols), int(ngrows), float(xgmin), float(ygmin),  jparams["grid-cellsize"])
    
    return np.savetxt(out_asc, zg, fmt='%f', header=header, comments='', delimiter=' ', 
                      newline='\n')

def create_grid(extent, resolution):
    """
    create meshgrid from extent and resolution of .las dataset
    """
    
    xmin = extent[0][0]
    ymin = extent[1][0]
    xmax = extent[0][1]
    ymax = extent[1][1]
    nrows = resolution[1]
    ncols = resolution[0]

    xi = np.linspace(xmin, xmax, num=ncols)
    yi = np.linspace(ymin, ymax, num=nrows)

    xg, yg = np.meshgrid(xi, yi)  # Generate grid
    
    return xg, yg

def idw_proc(xy, z, xg, yg, jparams):
    """
    iDW nn interpolation ~ https://stackoverflow.com/a/3119544
    """
    positions = np.vstack([xg.ravel(), yg.ravel()]).T

    invdisttree = Invdisttree(xy, z, leafsize=25, stat=1)
    zg = invdisttree(positions, nnear=jparams['idw-nn'], eps=0.1, p=jparams['idw-power'])
    zt = np.reshape(zg, xg.shape)
    
    return zt

def tin_proc(xy, z, xg, yg, jparams):
    """
    delaunay-based linear interpolation
    """
    delaunay = scipy.spatial.Delaunay(xy)
    interp  = LinearNDInterpolator(delaunay, z)
    zg = interp(xg, yg)
    #zt = np.flipud(zg)

    return zg, delaunay
    
def write_obj(pts, dt, obj_filename):
    """
    write .obj given points and scipy delaunay
    """
    print("=== Writing {} ===".format(obj_filename))
    f_out = open(obj_filename, 'w')
    pts = list(pts)
    for p in pts:
        f_out.write("v {:.2f} {:.2f} {:.2f}\n".format(p[0], p[1], p[2]))

    for simplex in dt.simplices:
        f_out.write("f {} {} {}\n".format(simplex[0]+1, simplex[1]+1, simplex[2]+1))
    f_out.close()

def clsy_pipe(las, jparams):
    """
    pdal pipeline to read .las, poisson resample, identify low noise, identify outliers, 
    classify ground and non-ground 
    ~ refine the classification with a nearest neighbor search and write the result as a .las
    """
    pline={
        "pipeline": [
            {
                "type": "readers.las",
                "filename": jparams['input-las']
            },
            {
                "type":"filters.sample",
                "radius": jparams['thinning-factor']
            },
            {
                "type":"filters.elm"
            },
            {
                "type":"filters.outlier",
                "method":"statistical",
                "mean_k": 12,
                "multiplier": 2.2
            },
            {
                "type":"filters.pmf",
                "slope": 0.2,
                "ignore":"Classification[7:7]",
                "initial_distance": jparams['gf-cellsize'],
                "max_distance": jparams['gf-distance'],
                "max_window_size": jparams['gf-cellsize'] + 10
            },
            {
                "type":"filters.range",
                "limits":"Classification[1:2]"
            },
            {
                "type" : "filters.neighborclassifier",
                "domain" : "Classification[2:2]",
                "k" : 7
            },
            {
                "type":"writers.las",
                "filename": las
            },
          ]
        } 
    
    pipeline = pdal.Pipeline(json.dumps(pline))
    #pipeline.validate() 
    count = pipeline.execute()
    array = pipeline.arrays[0]
    
    return array

def dsm_pipe(arr, extent, resolution, origin, name, jparams):
    """
    pdal pipelien to read a classified .las numpy (1 and 2, no noise[7]) 
    and generate a dsm ascii raster via writers.gdal built in idw
    """
    pline={
        "pipeline": [
            {
                "type":"writers.gdal",
                "filename": name,
                "resolution": jparams["grid-cellsize"],
                "radius": jparams["idw-nn"],
                "power": jparams["idw-power"],
                "window_size": 4,
                "output_type": "idw",
                "nodata": -9999,
                "dimension": "Z",
                "origin_x":  origin[0],
                "origin_y":  origin[1],
                "width":  resolution[0],
                "height": resolution[1]
            },
          ]
        } 
    
    p = pdal.Pipeline(json.dumps(pline), [arr])
    #pipeline.validate() 
    count = p.execute()
    #array = p.arrays[0]
          
    
def filter_ground(extent, resolution, origin, jparams, urban=False, idw=False, tin=False):
    """
    !!! TO BE COMPLETED !!!
    
    Function that reads a LAS file, performs thinning, then performs ground filtering, and creates a two rasters of the ground points. One with IDW interpolation and one with TIN interpolation.

      !!! You are free to subdivide the functionality of this function into several functions !!!
    
    Input:
        a dictionary jparams with all the parameters that are to be used in this function:
            - input-las:        path to input .las file,
            - thinning-factor:  thinning factor, ie. the `n` in nth point thinning method,
            - gf-cellsize:      cellsize for the initial grid that is computed as part of the 
            ground filtering algorithm,
            - gf-distance:      distance threshold used in the ground filtering algorithm,
            - gf-angle:         angle threshold used in the ground filtering algorithm,
            - idw-radius:       radius to use in the IDW interpolation,
            - idw-power:        power to use in the IDW interpolation,
            - output-las:       path to output .las file that contains your ground classification,
            - grid-cellsize:    cellsize of the output grids,
            - output-grid-tin:  filepath to the output grid with TIN interpolation,
            - output-grid-idw:  filepath to the output grid with IDW interpolation
            """
    
    if urban == True:
        las = jparams['outU-las']
        arr = clsy_pipe(las, jparams)
  
        ground = arr[(arr['Classification'] == 2) & (arr['Classification'] != 7) & (arr['Z'] <= 1)]
        grnd = np.vstack((ground['X'], ground['Y'], ground['Z'])).T

        xy = grnd[:,:2]
        gr_z = grnd[:,2]

        xg, yg = create_grid(extent, resolution)
        
        if idw == True:
        
            idw_zg = idw_proc(xy, gr_z, xg, yg, jparams)
            idw_asc = jparams['outU-grid-idw']
            write_ascii(idw_zg,  origin, resolution, jparams, idw_asc)
            
        if tin == True:
                        
            tin_zg, delaunay01 = tin_proc(xy, gr_z, xg, yg, jparams)
            tin_asc = jparams['outU-grid-tin']
            write_ascii(tin_zg, origin, resolution, jparams, tin_asc)
      
            write_obj(grnd, delaunay01, jparams['outU-surf-tin'])
            
        dsm_arr = arr[arr['Classification'] != 7]
        name = jparams["outU-dsm"]
        dsm_pipe(dsm_arr, extent, resolution, origin, name, jparams) 

    if urban == False:
        las = jparams['outR-las']
        arr = clsy_pipe(las, jparams)
  
        ground = arr[(arr['Classification'] == 2) & (arr['Classification'] != 7)]
        grnd = np.vstack((ground['X'], ground['Y'], ground['Z'])).T

        xy = grnd[:,:2]
        gr_z = grnd[:,2]

        xg, yg = create_grid(extent, resolution)
        
        if idw == True:
            idw_zg = idw_proc(xy, gr_z, xg, yg, jparams)
            idw_asc = jparams['outR-grid-idw']
            write_ascii(idw_zg,  origin, resolution, jparams, idw_asc)
      
        if tin == True:
            tin_zg, delaunay01 = tin_proc(xy, gr_z, xg, yg, jparams)
          
            tin_asc = jparams['outR-grid-tin']
            write_ascii(tin_zg, origin, resolution, jparams, tin_asc)
          
            write_obj(grnd, delaunay01, jparams['outR-surf-tin'])
    
        dsm_arr = arr[arr['Classification'] != 7]
        name = jparams["outR-dsm"]
        dsm_pipe(dsm_arr, extent, resolution, origin, name, jparams) 