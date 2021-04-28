#-- my_code_hw01.py
#-- hw01 GEO1015/2018
#-- [YOUR NAME]
#-- [YOUR STUDENT NUMBER] 
"""#-- ['arkriger']"""
#-- [YOUR STUDENT NUMBER] 

"""
Basic terrain surface interpolation - Nearest Neigbours (NN), Inverse Distance Weighing (IDW), 
Delaunay ~ Triangulated Irregular Network (TIN) and Kriging (or GaussianProcessRegressor). 
"""
import os
import math
import numpy as np

import scipy.spatial
from scipy.interpolate import NearestNDInterpolator

from invdisttree import Invdisttree
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import seaborn as sns

import time

def create_grid(xy, pixel):
    
    xmin = min(x[0] for x in xy)
    ymin = min(x[1] for x in xy)
    xmax = max(x[0] for x in xy)
    ymax = max(x[1] for x in xy)
    nrows = int((ymax - ymin)/pixel)
    ncols = int((xmax - xmin)/pixel)
    
    xi = np.linspace(xmin, xmax, num=ncols)
    yi = np.linspace(ymin, ymax, num=nrows)
    
    xg, yg = np.meshgrid(xi, yi)  # Generate grid
    
    return xg, yg

def write_ascii(xg, yg, zg, pixel, out_asc):
    
    zg[np.isnan(zg)] = -9999

    xgmin = min(xg.flatten())
    ygmin = min(yg.flatten())
    xgmax = max(xg.flatten())
    ygmax = max(yg.flatten())
    ngrows = int((ygmax - ygmin)/pixel)
    ngcols = int((xgmax - xgmin)/pixel)
       
    zg = np.flipud(zg)
     
    # ASCII file header
    header = "ncols %d \nnrows %d\nxllcorner %f\nyllcorner %f\ncellsize %f\nnodata_value -9999" \
            % (int(ngcols), int(ngrows), float(xgmin), float(ygmin),  pixel)
    
    return np.savetxt(out_asc, zg, fmt='%f', header=header, comments='', delimiter=' ', newline='\n')

def plot(xg, yg, zg, tin=False, title = None):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    
    plt.pcolormesh(zg.transpose(), cmap=plt.cm.jet)
    plt.xlim([0, zg.shape[0]])
    plt.ylim([0, zg.shape[1]])
    plt.colorbar()
    plt.title(title)
    
    ax = fig.add_subplot(1,2,2, projection='3d')
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    
    if tin == True:
        b = np.where(np.isnan(zg), 0, zg)
        my_col = cm.jet(b/np.amax(b))
        ax.plot_surface(xg, yg, zg, rstride=1, cstride=1, facecolors = my_col, 
                linewidth=0, antialiased=True)
    else:
        ax.plot_surface(xg, yg, zg, rstride=1, cstride=1, cmap=plt.cm.jet,
                        linewidth=0, antialiased=True)
    plt.show()                                

def nn_interpolation(list_pts_3d, j_nn):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with nearest neighbour interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_nn:        the parameters of the input for "nn"
    Output:
        returns the value of the area
 
    """  
    print("=== Nearest neighbour interpolation ===")
    start = time.time()

    # print("cellsize:", j_nn['cellsize'])

    #-- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # kd = scipy.spatial.KDTree(list_pts)
    # d, i = kd.query(p, k=1)
    
    xy = [l[:2] for l in list_pts_3d]
    z = [l[-1] for l in list_pts_3d]
    
    pixel = j_nn['cellsize']
    
    xg, yg = create_grid(xy, pixel)  # Generate grid
    
    # Interpolate with NN with KDTree parameters
    interp = NearestNDInterpolator(xy, z, tree_options={'leafsize':25,
                                                        'copy_data':True})
    zg = interp(xg, yg)
    plot(xg, yg, zg, tin=False, title = 'Nearest Neighbour')
    out_asc = j_nn['output-file']
    
    write_ascii(xg, yg, zg, pixel, out_asc)
    
    print("File written to", j_nn['output-file'])
    end = time.time()
    print('runtime:', end - start)

def idw_interpolation(list_pts_3d, j_idw):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with IDW
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_idw:       the parameters of the input for "idw"
    Output:
        returns the value of the area
 
    """  
    print("=== IDW interpolation ===")
    start = time.time()

    # print("cellsize:", j_idw['cellsize'])
    # print("radius:", j_idw['radius'])

    #-- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # kd = scipy.spatial.KDTree(list_pts)
    # i = kd.query_ball_point(p, radius)
    
    xy = np.array([l[:2] for l in list_pts_3d])
    z = np.array([l[-1] for l in list_pts_3d])
    
    pixel = j_idw['cellsize']
    
    xg, yg = create_grid(xy, pixel)  # Generate grid
    
    positions = np.vstack([xg.ravel(), yg.ravel()]).T
    
    # IDW with KDTree
    invdisttree = Invdisttree(xy, z, leafsize=25, stat=1)
    zg = invdisttree(positions,
                     nnear=j_idw['radius'], eps=0.1, p=j_idw['power'])
    
    zt = np.reshape(zg, xg.shape)
    plot(xg, yg, zt, tin=False, title = 'Inverse Distance Weighting')
    out_asc = j_idw['output-file']
    # ASCII file 
    write_ascii(xg, yg, zt, pixel, out_asc)
    
    print("File written to", j_idw['output-file'])
    end = time.time()
    print('runtime:', end - start) 


def tin_interpolation(list_pts_3d, j_tin):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with linear in TIN interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_tin:       the parameters of the input for "tin"
    Output:
        returns the value of the area
 
    """  
    print("=== TIN interpolation ===")
    start = time.time()

    #-- example to construct the DT
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay
    # dt = scipy.spatial.Delaunay([])
    
    xy = np.array([l[:2] for l in list_pts_3d])
    z = np.array([l[-1] for l in list_pts_3d])
    
    pixel = j_tin['cellsize']
    
    xg, yg = create_grid(xy, pixel)  # Generate grid
    
    delaunay = scipy.spatial.Delaunay(xy)
    interp  = scipy.interpolate.CloughTocher2DInterpolator(delaunay, z)
    zg = interp(xg, yg)
    plot(xg, yg, zg, tin=True, title = 'Delaunay_TIN')
    out_asc = j_tin['output-file']
    # ASCII file 
    write_ascii(xg, yg, zg, pixel, out_asc) 
    
    print("File written to", j_tin['output-file'])
    end = time.time()
    print('runtime:', end - start) 

def kriging_interpolation(list_pts_3d, j_kriging):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with ordinary kriging interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_kriging:       the parameters of the input for "kriging"
    Output:
        returns the value of the area
 
    """  
    print("=== Ordinary kriging interpolation ===")
    start = time.time()
    xy = np.array([l[:2] for l in list_pts_3d])
    z = np.array([l[-1] for l in list_pts_3d])
    
    pixel = j_kriging['cellsize']

    xg, yg = create_grid(xy, pixel)  # Generate grid
    kernel = RBF(length_scale=pixel)
    gp = GaussianProcessRegressor(normalize_y=True,
                                  alpha = 0.1,  
                                  kernel=kernel)
    
    gp.fit(xy, z)
    X_grid = np.stack([xg.ravel(), yg.ravel()]).T
    zg = gp.predict(X_grid).reshape(xg.shape)                      
    #zt = np.reshape(field, xg.shape)
    plot(xg, yg, zg, tin=False, title = 'Kriging')
    out_asc = j_kriging['output-file']
    # ASCII file 
    write_ascii(xg, yg, zg, pixel, out_asc)

    print("File written to", j_kriging['output-file'])
    end = time.time()
    print('runtime:', end - start) 
