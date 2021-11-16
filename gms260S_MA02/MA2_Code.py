# -*- coding: utf-8 -*-
# env/AHN3
# created on Sun Nov 14 18:09:38 2021

# author: arkriger
# GMS260S. Geomatics 2
# Cape Peninsula University of Technology
# Engineering and the Built Environment
# Civil Engineering and Surveying

# - takes the xyz.csv and executes MA2_Code.py; that is:
# - enter PDAL pipeline (outlier removal, 
#                        ground filtering, write .las,
#                        raster DTM via TIN with Laplace interpolation, 
#                        raster DSM quadrant IDW,
#                        hillshade and 
#                        contours)

# - dtm and dsm courtesy AHN3 procedure: 
#     https://github.com/tudelft3d/geo1101.2020.ahn3 and 
#     https://github.com/khalhoz/geo1101-ahn3-GF-and-Interpolation 

import os
import glob

import math
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from laspy.file import File

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pdal

from osgeo import gdal, osr, ogr

def get_csv(fpath, jparams):
    
    # Takes the filepath to an input csv file and the
    # desired output raster cell size. Reads the ply file, performs basic cleaning, 
    # reprojects, ground filtering, transforms to a 
    # .las and outputs the ground points as a numpy array. 
    # Also establishes some basic raster parameters:
    #     - the extents
    #     - the resolution in coordinates
    #     - the coordinate location of the relative origin (bottom left)
    
    #Import LAS into numpy array 
    array = clsy_pipe(fpath, jparams)
    lidar_points = np.array((array['X'], array['Y'], array['Z'])).transpose()
    
    #Transform to pandas DataFrame
    lidar_df = pd.DataFrame(lidar_points, columns=['x', 'y', 'z'])
    extent = [[lidar_df.x.min(), lidar_df.x.max()],
              [lidar_df.y.min(), lidar_df.y.max()]]
    res = [math.ceil((extent[0][1] - extent[0][0]) / jparams["size"]),
           math.ceil((extent[1][1] - extent[1][0]) / jparams["size"])]
    origin = [np.mean(extent[0]) - (jparams["size"] / 2) * res[0],
              np.mean(extent[1]) - (jparams["size"] / 2) * res[1]]
    ul_origin = [np.mean(extent[0]) - (jparams["size"] / 2) * res[0],
                 np.mean(extent[1]) + (jparams["size"] / 2) * res[1]]
    
    # if gnd_only == True:
    #     array = array[(array['Classification'] == 2) & (array['Classification'] != 7)]
    #     #in_np = np.vstack((ground['X'], ground['Y'], ground['Z'])).T
    # else:
    #     # don't .T here. 
    #     # ~ .T inside the various functions so that pdal_idw does not have to read the file
    #     array = array[array['Classification'] != 7]
        
    return array, res, origin, ul_origin

def clsy_pipe(csv, jparams):

    # pdal pipeline to read .las, poisson resample, identify low noise, identify outliers, 
    # classify ground and non-ground 
    # ~ refine the classification with a nearest neighbor search and write the result as a .las
   
    pline={
        "pipeline": [
            {
                "type": "readers.text",
                "filename": csv,
                "default_srs": jparams['crs'],
            },
            # {
            #     "type":"filters.sample",
            #     "radius": jparams['thinning-factor']
            # },
            # {
            #     "type":"filters.reprojection",
            #     "in_srs":"+proj=geocent +ellps=WGS84 +datum=WGS84 +no_defs",
            #     "out_srs": jparams['crs']
            # },
            {
                "type":"filters.elm"
            },
            {
                "type":"filters.outlier",
                "method":"statistical",
                "mean_k": 12,
                "multiplier": 1.2
            },
            {
                "type":"filters.pmf",
                #"slope": jparams["gf-angle"],
                "ignore":"Classification[7:7]",
                "initial_distance": jparams['initial_distance'],
                "max_distance": jparams['max_distance'],
                "max_window_size": jparams['initial_distance'] + 20
            },
            {
                "type":"filters.range",
                "limits":"Classification[1:2]"
            },
            # {
            #     "type" : "filters.neighborclassifier",
            #     "domain" : "Classification[2:2]",
            #     "k" : 7
            # },
            # {
            #      "type": "filters.approximatecoplanar",
            #      "knn": 10
            # },
            # {
            #     "type":"filters.estimaterank",
            # },
            {
                "type":"writers.las",
                "filename": jparams['out-las']
            }
            # {
            #     "type":"filters.crop",
            #     "bounds": jparams['bounds']
            # }
          ]
        } 
    
    pipeline = pdal.Pipeline(json.dumps(pline))
    #pipeline.validate() 
    count = pipeline.execute()
    array = pipeline.arrays[0]
    
    return array

def execute_startin(array, res, origin, size, method):
    
    # Takes the grid parameters and the ground points. Interpolates
    # either using the TIN-linear or the Laplace method. Uses a
    # -9999 no-data value. Fully based on the startin package.
    
    import startinpy
    
    array = array[(array['Classification'] == 2) & (array['Classification'] != 7)]#\
                  #& (array['Coplanar'] != 0) & (array['Rank'] != 3)]
    pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    tin = startinpy.DT(); tin.insert(pts)
    ras = np.zeros([res[1], res[0]])
    if method == 'startin-TINlinear':
        def interpolant(x, y): return tin.interpolate_tin_linear(x, y)
    elif method == 'startin-Laplace':
        def interpolant(x, y): return tin.interpolate_laplace(x, y)
    yi = 0
    for y in np.arange(origin[1], origin[1] + res[1] * size, size):
        xi = 0
        for x in np.arange(origin[0], origin[0] + res[0] * size, size):
            tri = tin.locate(x, y)
            if tri != [] and 0 not in tri:
                ras[yi, xi] = interpolant(x, y)
            else: ras[yi, xi] = -9999
            xi += 1
        yi += 1
    return ras, tin


def execute_idwquad(array, res, origin, size,
                    start_rk, pwr, minp, incr_rk, method, tolerance, maxiter):
    
    # Creates a KD-tree representation of the tile's points and
    # executes a quadrant-based IDW algorithm on them. Although the
    # KD-tree is based on a C implementation, the rest is coded in
    # pure Python (below). Keep in mind that because of this, this
    # is inevitably slower than the rest of the algorithms here.
    # To optimise performance, one is advised to fine-tune the
    # parametrisation, especially tolerance and maxiter.
    # More info in the GitHub readme.
    
    from scipy.spatial import cKDTree
    pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    ras = np.zeros([res[1], res[0]])
    tree = cKDTree(np.array([pts[:,0], pts[:,1]]).transpose())
    yi = 0
    for y in np.arange(origin[1], origin[1] + res[1] * size, size):
        xi = 0
        for x in np.arange(origin[0], origin[0] + res[0] * size, size):
            done, i, rk = False, 0, start_rk
            while done == False:
                if method == "radial":
                    ix = tree.query_ball_point([x, y], rk, tolerance)
                elif method == "k-nearest":
                    ix = tree.query([x, y], rk, tolerance)[1]
                xyp = pts[ix]
                qs = [
                        xyp[(xyp[:,0] < x) & (xyp[:,1] < y)],
                        xyp[(xyp[:,0] > x) & (xyp[:,1] < y)],
                        xyp[(xyp[:,0] < x) & (xyp[:,1] > y)],
                        xyp[(xyp[:,0] > x) & (xyp[:,1] > y)]
                     ]
                if min(qs[0].size, qs[1].size,
                       qs[2].size, qs[3].size) >= minp: done = True
                elif i == maxiter:
                    ras[yi, xi] = -9999; break
                rk += incr_rk; i += 1
            else:
                asum, bsum = 0, 0
                for pt in xyp:
                    dst = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                    u, w = pt[2], 1 / dst ** pwr
                    asum += u * w; bsum += w
                    ras[yi, xi] = asum / bsum
            xi += 1
        yi += 1
    return ras

def write_geotiff(raster, origin, size, crs, fpath):
    
    # Writes the interpolated TIN-linear and Laplace rasters
    # to disk using the GeoTIFF format. The header is based on
    # the raster array and a manual definition of the coordinate
    # system and an identity affine transform.
    
    import rasterio
    from rasterio.transform import Affine
    transform = (Affine.translation(origin[0], origin[1])
                 * Affine.scale(size, size))
    #raster  = np.flip(raster, 0)
    with rasterio.Env():
        with rasterio.open(fpath, 'w', driver = 'GTiff',
                           height = raster.shape[0],
                           width = raster.shape[1],
                           count = 1,
                           dtype = rasterio.float32,
                           crs = crs,
                           #crs = '+proj=utm +zone=3 +ellps=WGS84 +datum=WGS84 +units=m +no_defs',
                           transform = transform
                           ) as out_file:
            out_file.write(raster.astype(rasterio.float32), 1)

def pdal_idw(array, res, origin, name, jparams):
    
    # Sets up a PDAL pipeline that reads a ground filtered LAS
    # file, and writes it via GDAL. The GDAL writer has interpolation
    # options, exposing the radius, power and a fallback kernel width
    # to be configured.
    
    import pdal
    pline={
        "pipeline": [
            {
                "type":"writers.gdal",
                "filename": name + '_idwPDAL.tif',
                "resolution": jparams['size'],
                "radius": jparams['pdal-idw-rad'],
                "power": jparams['pdal-idw-pow'],
                "window_size": jparams['pdal-idw-wnd'],
                "output_type": "idw",
                "nodata": -9999,
                "dimension": "Z",
                "origin_x":  origin[0],
                "origin_y":  origin[1],
                "width":  res[0],
                "height": res[1]
                #"out_srs": jparams['crs']
            },
          ]
        } 
    
    p = pdal.Pipeline(json.dumps(pline), [array])
    #pipeline.validate() 
    p.execute()    
    
def plot(jparams):
    files = glob.glob(os.path.join('*.tif'))
    img_l = []
    #char2 = '.asc'
    for file in files:
        ds = gdal.Open(file)#.ReadAsArray()
        array = ds.ReadAsArray()
        array  = np.flip(array, 0)
        nan_array = array
        nan_array[array == -9999] = np.nan
        img_l.append(nan_array)
        x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
        # close dataset
        ds = None
        
    #gdalArray = ds.ReadAsArray()
    #to plot the image in the right location with matplotlib you'll need the extent
    nrows, ncols = array.shape
    # if the image isn't rotated/skewed/etc. 
    # This is not the correct method in general, but let's ignore that for now
    # If dxdy or dydx aren't 0, then this will be incorrect
    #x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows
    extent=[x0, x1, y1, y0]
        
    #fig, ax = plt.subplots(1, 2, figsize=(14,6))#, sharey=True)
    fig, (ax, ax1) = plt.subplots(1, 2, figsize=(14, 8), sharex=True)

    ax = plt.subplot(131)
    cf = plt.imshow(img_l[0], cmap=plt.cm.jet, extent=extent)
    ax.set_title('DSM_idwQUAD', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    
    #img_l[1]  = np.flip(img_l[1], 0)
    ax1 = plt.subplot(132)#, sharex = ax)
    cb = plt.imshow(img_l[1], cmap=plt.cm.plasma, extent=extent)
    ax1.set_title('DTM_laPlace', fontdict={'fontsize': 13, 'fontweight': 'medium'})
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cb, cax=cax)
    #plt.colorbar()
    fig.tight_layout()
    plt.show()
    
def do_Hillshade(jparams):
    
    ds = gdal.Open('efoto_xyz_dtm_tinLaplace.tif')
    band = ds.GetRasterBand(1)
    gdalArray = band.ReadAsArray()
    
    #to plot the image in the right location with matplotlib you'll need the extent
    nrows, ncols = gdalArray.shape
    # if the image isn't rotated/skewed/etc. 
    # This is not the correct method in general, but let's ignore that for now
    # If dxdy or dydx aren't 0, then this will be incorrect
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows
    extent=[x0, x1, y1, y0]
    
    hill = gdal.DEMProcessing('', 'efoto_xyz_dtm_tinLaplace.tif', 'hillshade', azimuth=35,
                              altitude=135, zFactor=1.5, alg='Horn',
                              format = 'MEM', addAlpha=True)
    band = hill.GetRasterBand(1)
    array = band.ReadAsArray()
    array  = np.flip(array, 0)
    #array[array == -9999] = np.nan
    #plot_dem(array, cmap='Greys')#, hill=True)
    
    plt.figure(figsize=(11,11))
    ax = plt.gca()
    #cmap=cmap
    #x0, x1, y0, y1 = x0, x1, y0, y1
    cb = ax.imshow(array, cmap='Greys', extent=extent)
    #rd.rdShow(image, axes=True, cmap=cmap, figsize=(9, 9))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cb, cax=cax)
    plt.show()
    
    return array
    
def do_Contours(jparams, hillshade):
    
    ds = gdal.Open('efoto_xyz_dtm_tinLaplace.tif')
    band = ds.GetRasterBand(1)
    gdalArray = band.ReadAsArray()
    array = ds.ReadAsArray()
    nan_array = array
    nan_array[array == -9999] = np.nan
    _dem  = np.flip(nan_array, 0)
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    
    #to plot the image in the right location with matplotlib you'll need the extent
    nrows, ncols = gdalArray.shape
    # if the image isn't rotated/skewed/etc. 
    # This is not the correct method in general, but let's ignore that for now
    # If dxdy or dydx aren't 0, then this will be incorrect
    x0, dx, dxdy, y0, dydx, dy = ds.GetGeoTransform()
    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows
    extent=[x0, x1, y1, y0]

    #define not a number
    demNan = -9999
    
    contourPath = 'efoto_contours.shp'
    contourDs = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(contourPath)
    
    #define layer name and spatial 
    contourShp = contourDs.CreateLayer('contour', proj)
    #define fields of id and elev
    fieldDef = ogr.FieldDefn("ID", ogr.OFTInteger)
    contourShp.CreateField(fieldDef)
    fieldDef = ogr.FieldDefn("elev", ogr.OFTReal)
    contourShp.CreateField(fieldDef)
    
    def roundup(x):
        return int(math.ceil(x / 10.0)) * 10
    
    #define number of contours and range ~ 50m
    demMax = roundup(gdalArray.max())
    demMin = roundup(gdalArray[gdalArray != demNan].min())
    conNum = jparams["cnt_interval"]
    conList =[int(x) for x in np.linspace(demMin, demMax, conNum)]
    
    #Write shapefile using noDataValue
    #ContourGenerate(Band srcBand, double contourInterval, double contourBase, 
    #                int fixedLevelCount, int useNoData, double noDataValue, 
    #                Layer dstLayer, int idField, int elevField)
    gdal.ContourGenerate(band, 0, 0, conList, 1, demNan, 
                         contourShp, 0, 1)
    
    ds = None
    contourDs.Destroy()
    
    cnt = gpd.read_file('efoto_contours.shp')
       
    plt.figure(figsize=(11,11))
    ax = plt.gca()
    #cmap=cmap
    cb = ax.imshow(_dem, cmap='terrain', zorder=1, alpha=0.5, extent=extent)#[x0, x1, y1, y0])
    #ax.imshow(hillshade, cmap='Greys', zorder=1, alpha=0.5, extent=extent)#[x0, x1, y1, y0])
    cnt.plot(ax=ax, facecolor='none', edgecolor='r', linewidth=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cb, cax=cax)
    plt.show()
    
