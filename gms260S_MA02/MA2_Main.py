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
#                        ground filtering, write .las)
# - raster DTM via TIN with Laplace interpolation;
# - raster DSM quadrant IDW;
# - hillshade and 
# - contours.

# - dtm and dsm courtesy AHN3 procedure: 
#     https://github.com/tudelft3d/geo1101.2020.ahn3 and 
#     https://github.com/khalhoz/geo1101-ahn3-GF-and-Interpolation 

import os 
import glob
import json
import numpy as np
from pathlib import Path

import startinpy
#import rasterio

import time
from datetime import timedelta

from MA2_Code import execute_startin, get_csv, execute_idwquad, write_geotiff, pdal_idw, plot, do_Contours, do_Hillshade

def main():
       
    start = time.time()
    
    jparams = json.load(open('params.json'))
    
    infile = jparams["input-ply"]
    array, res, origin, ul_origin = get_csv(infile, jparams)
    
    if jparams["dtm"] == "True":
        name = Path(infile).stem + '_dtm'
        
        rasLap, tinLap = execute_startin(array, res, origin, jparams["size"], 
                                         method='startin-Laplace')
        write_geotiff(rasLap, origin, jparams["size"], jparams["crs"], name + '_tinLaplace.tif')
        tinLap.write_obj(name + '_TINlaplace.obj')
        
        #plot()
        
    if jparams["dsm"] == "True":
        name = Path(infile).stem + '_dsm'
        
        ras = execute_idwquad(array, res, origin, jparams["size"],
                              jparams["start_rk"], jparams["pwr"], jparams["minp"], 
                              jparams["incr_rk"], jparams["method"], jparams["tolerance"], 
                              jparams["maxiter"])
        write_geotiff(ras, origin, jparams["size"], jparams["crs"], name + '_idwQUAD.tif')
        
        # array  = np.flip(array, 0)
        # array = array[array['Classification'] != 7]
        # pdal_idw(array, res, origin, name, jparams)
        
    if jparams["plot"] == "True":
        plot(jparams)
        
    if jparams["hillshade"] == "True":
        hillshade = do_Hillshade(jparams)
    
    if jparams["contours"] == "True":
        do_Contours(jparams, hillshade)
        
    #-- timeit
    end = time.time()
    print('runtime:', str(timedelta(seconds=(end - start))))
    
if __name__ == "__main__":
    main()