#-- my_code_hw02.py
#-- hw02 GEO1015/2018
#-- [arkriger]
#-- [YOUR STUDENT NUMBER] 
#-- [YOUR NAME]
#-- [YOUR STUDENT NUMBER] 

"""
take an input .tif, extract .xyz and refine a TIN based on a threshold 
~ (place the point with the maximum vertical error) when the error is less than the 
threshold stop. Output .obj, .tif, text_xyz of important_pts, xyz of the entire interpolated 
surface [equivilant to the .tif], difference raster 
and diff_csv [original_xyz minus threshold TIN at same xy]
"""
import os
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
from osgeo import gdal, osr

from scipy.spatial import ConvexHull
from  scipy.spatial import Delaunay
import scipy.interpolate

import time
from datetime import timedelta

### !!!!!!!!  ######## ~~ this is important
gdl = 'C:/{path-to-gdal_calc.py}/Scripts/'
### !!!!!!!!  ######## ~~ this is important

def GetGeoInfo(FileName):
    """extract information from .tif"""
    SourceDS = gdal.Open(FileName, gdal.GA_ReadOnly)
    #NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    
    xmin, xpixel, _, ymax, _, ypixel =  SourceDS.GetGeoTransform()
    xmax = xmin + xsize * xpixel
    ymin = ymax + ysize * ypixel
    
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    #DataType = SourceDS.GetRasterBand(1).DataType
    #DataType = gdal.GetDataTypeName(DataType)

    SourceDS = None

    return xmin, ymin, xmax, ymax, xpixel, ypixel, xsize, ysize, Projection

def create_grid(jparams):
    """create a meshgrid based on the parameters (extent, size, resolution, etc.) 
    from the original dataset
    """
    xmin, ymin, xmax, ymax, xpixel, ypixel, xsize, ysize, Projection = GetGeoInfo(jparams['input-file'])

    """ xmin, xpixel, _, ymax, _, ypixel = ds.GetGeoTransform()
        xmax = xmin + width * xpixel
        ymin = ymax + height * ypixel
    """
    xi = np.linspace(xmin, xmax, num=xsize)#ncols)
    yi = np.linspace(ymin, ymax, num=ysize)#nrows)

    xg, yg = np.meshgrid(xi, yi, indexing='xy')  # Generate grid
    return xg, yg

def write_grid(xg, yg, zg, jparams):
    """write a meshgrid [interpolated surface] to xyz
    """
    print('=== writing mesh surface to xyz ===')
    
    data = np.stack([xg.ravel(), yg.ravel(), zg.ravel()], axis=1)    #T.
    np.savetxt(jparams['output-file-grid'], data, delimiter=' ', header="x y z", 
                  comments="")

def write_tif(zg, jparams):
    """
    writes a .tif based on the [refined] interpolated surface (an array) while harvesting 
    the parameters (size, resolution, projection, etc) from the original dataset.
    """
    print('=== writing .tif ===')
    xmin, ymin, xmax, ymax, xpixel, ypixel, xsize, ysize, Projection = GetGeoInfo(jparams['input-file'])
      
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(jparams['output-file-tif'],
                                  xsize, ysize, 1, gdal.GDT_Float32) 
    #
    geotrans = (xmin, xpixel, 0, ymax, 0, ypixel) 
    output_raster.SetGeoTransform(geotrans)
    output_raster.SetProjection(Projection.ExportToWkt())
    
    output_raster.GetRasterBand(1).WriteArray(zg)

    output_raster.FlushCache() ##saves to disk!!
    output_raster = None
        
def remove_noData(jparams):
    """remove NoData from xyx"""
    df = pd.read_csv(jparams['output-xyz'],
                     delimiter= ' ',
                     names = ['x', 'y', 'z'])
    df.apply(pd.to_numeric)
    df = df[df.z != -32768]
    df = np.round(df, 3)
    df.to_csv(jparams['output-NoNaNxyz'], index=False, sep=' ')
    
def read_pts_from_grid(jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that reads a grid in .tif format and retrieves the pixels as a list of (x,y,z) points shifted to the origin
     
    Input from jparams:
        input-file:  a string containing the path to a grid file with elevations
    Returns:
        a numpy array where each row is one (x,y,z) point. Each pixel from the grid gives one point (except for no-data pixels, these should be skipped).
    """
    print("=== Reading points from grid ===")

    # Tip: the most efficient implementation of this function does not use any loops. Use numpy functions instead.
    
    kwargs = {
    'format': 'xyz',
    'noData': -32768,
    }
    ds = gdal.Translate(jparams['output-xyz'], jparams['input-file'], **kwargs)
    ds = None # close and save
    
    remove_noData(jparams)

def simplify_by_refinement(jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that takes a list of points and constructs a TIN with as few points as possible, 
    while still satisfying the error-threshold. 

    This should be an implemented as a TIN refinement algorithm using greedy insertion. 
    As importance measure the vertical error should be used.
     
    Input:
        pts:             a numpy array with on each row one (x,y,z) point
        from jparams:
        error-threshold: a float specifying the maximum allowable vertical error in the TIN
    Returns:
        a numpy array that is a subset of pts and contains the most important points with 
        respect to the error-threshold
    """
    print("=== TIN simplification ===")

    # Remember: the vertices of the initial TIN should not be returned
    start = time.time()
      
    ## ~~ read the xyz and refine the TIN
    f = jparams['output-NoNaNxyz']
    df = pd.read_csv(f, delimiter=' ', header='infer')
    
    xy = list(zip(df.x, df.y))
    z = df['z'].tolist()
    
    cv = ConvexHull(xy, qhull_options='Qt')
    boundary = [xy[i] for i in cv.vertices]
    removing_xy = [i for j, i in enumerate(xy) if j not in cv.vertices]
    #[del xy[i] for i in cv.vertices]
    new_z = [z[i] for i in cv.vertices]
    adding_z = new_z
    removing_z = [i for j, i in enumerate(z) if j not in cv.vertices]
    #initial triangulation
    delaunay = Delaunay(np.array(boundary), incremental=True,
                        qhull_options='QJ')
    
    max_err = 1000000
    
    while max_err > jparams['error-threshold']:
        err = []
        for i, n in enumerate(removing_xy):
            #triangulation without point
            tri = Delaunay(np.array(boundary), incremental=True,
                        qhull_options='QJ')
            #what is the height of the interpolated surface?
            interp2  = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                    np.array(adding_z))
            test1 = interp2(np.array(n))
            
            adding_z.append(removing_z[i])
            #triangulation with point
            tri.add_points([list(n)], restart=False)
            #what is the height of the interpolated surface?
            tri_interp  = scipy.interpolate.CloughTocher2DInterpolator(tri,
                                                                        np.array(adding_z))        
            test = tri_interp(np.array(n))
            #what is the difference between the two surfaces at the point
            err.append(test1 - test)
            del adding_z[-1]
 
        max_value = max(err, key=abs)
        max_err = abs(max_value)
        max_index = err.index(max_value)
        boundary.append(removing_xy[max_index])
        
        adding_z.append(removing_z[max_index]) 
        del removing_z[max_index]
        delaunay.add_points([list(removing_xy[max_index])], restart=False)
        del removing_xy[max_index]
     
    delaunay.close()
    x = [x[0] for x in boundary]
    y = [x[1] for x in boundary]
    pts = list(zip(x, y, adding_z))
    pts_important = [list(elem) for elem in pts]
    df = pd.DataFrame(pts_important)
    df.to_csv(jparams['output-file-impXYZ'], index=False, sep=' ', 
              header=['x', 'y', 'z'])
    
    interp  = scipy.interpolate.CloughTocher2DInterpolator(delaunay, 
                                                            np.array(adding_z)) 
    #interpolate
    xg, yg = create_grid(jparams)
    zg = interp(xg, yg)
    zg = np.flipud(zg)
    
    #write interpolated surface to csv
    write_grid(xg, yg, zg, jparams)
    
    #create image 
    write_tif(zg, jparams)
    
    end = time.time()
    print('runtime:', str(timedelta(seconds=(end - start))))
    return pts_important, interp, delaunay

def compute_differences(interp, jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that computes the elevation differences between the input grid and the Delaunay triangulation that is constructed from pts_important. 
    The differences are computed for each pixel of the input grid by subtracting the grid elevation from the TIN elevation. 
    The output is a new grid that stores these differences as float32 and has the same width, height, transform, crs and nodata value as the input grid.

    Input:
        pts_important:          numpy array with the vertices of the simplified TIN
        from jparams:
            input-file:                 string that specifies the input grid
            output-file-differences:    string that specifies where to write the output grid file with the differences
    """
    print("=== Computing differences ===")
    
    ## ~~ read the xyz
    f = jparams['output-NoNaNxyz']
    df = pd.read_csv(f, delimiter=' ', header='infer')
    
    xy = list(zip(df.x, df.y))
    z = df['z'].tolist()
    #interpolate at original locations
    gr = interp(np.array(xy))
    
    C = [a - b for a, b in zip(gr, z)]
    diff = list(zip(df.x, df.y, C))
    
    df = pd.DataFrame(diff)
    op = jparams['output-file-diffxyz']
    df.to_csv(op, index=False, sep=' ', header=['x', 'y', 'z']) 

    """
    additionally a difference-raster can be created
    """
    gclc = os.path.join(gdl, 'gdal_calc.py')
    #expression
    calc_expr = '"A-B"'
    nodata = '-9999'
    
    gdal_calc_str = 'python {0} -A {1} -B {2} --outfile={3} --calc={4} --NoDataValue={5}'
    gdal_calc_process = gdal_calc_str.format(gclc, 
                                             jparams['input-file'], 
                                             jparams['output-file-tif'], 
                                             jparams['output-file-diffTiff'], 
                                             calc_expr, nodata)#, typeof)

    # Call process.
    subprocess.check_call(gdal_calc_process)

