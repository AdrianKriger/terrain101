# -*- coding: utf-8 -*-
"""
Created on Sat May 22 18:10:36 2021

@author: arkriger
"""
import os 
import glob
import json
import numpy as np
from pathlib import Path

import startinpy
import rasterio

from code_AHN3_local import execute_startin, write_asc, get_las, pdal_idw, execute_idwquad, \
    execute_cgal, execute_cgal_cdt, basic_flattening, write_geotiff

def main():
    """
    - takes the .las from params_local.json, executes code_AHN3_local.py for each interpolation 
    technique.
    """
    jparams = json.load(open('params_local.json'))
    for method in jparams['interpolation_methods']:
        # Gets current working directory
        path = os.getcwd()
        # Joins the folder that we wanted to create
        folder_name = method
        path = os.path.join(path, folder_name) 
        # Creates the folder, and checks if it is created or not.
        os.makedirs(path, exist_ok=True)
        
    infile = jparams["input-las"]
    
    if jparams["dtm"] == "True":
        array, res, origin, ul_origin = get_las(infile, jparams["size"], gnd_only = True)
        name = Path(infile).stem + '_dtm'
        
        rasLin, tinLin = execute_startin(array, res, origin, jparams["size"], 
                                      method='startin-TINlinear')
        tinflat, raspolys = basic_flattening(rasLin, res, origin, jparams["size"], 
                                              tin = tinLin)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tinLinear"] + name + '_tinLinear.tif')
        # tinLin.write_obj(jparams["tinLinear"] + name + '_TINlinear.obj')
    
        rasLap, tinLap = execute_startin(array, res, origin, jparams["size"], 
                                      method='startin-Laplace')
        tinflat, raspolys = basic_flattening(rasLap, res, origin, jparams["size"], 
                                              tin = tinLap)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tinLaplace"] + name + '_tinLaplace.tif')
        tinLap.write_obj(jparams["tinLaplace"] + name + 'TINlaplace.obj')
    
        pdal_idw(array, res, origin, name, jparams)
        ds = rasterio.open(jparams['pdal-idw'] + name + '_idwPDAL.tif')#.ReadAsArray()
        img = ds.read(1)
        img  = np.flip(img, 0)
        #img[img == -9999] = np.nan
        ds = None
        tinflat, raspolys = basic_flattening(img, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["pdal-idw"] + name + '_idwPDAL.tif')

        ras = execute_idwquad(array, res, origin, jparams["size"],
                    jparams["start_rk"], jparams["pwr"], jparams["minp"], 
                    jparams["incr_rk"], jparams["method"], jparams["tolerance"], 
                    jparams["maxiter"])
        tinflat, raspolys = basic_flattening(ras, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["quad-idw"] + name + '_idwQUAD.tif')
        
        ras, tinCGAL = execute_cgal(array, res, origin, jparams["size"])
        tinflat, raspolys = basic_flattening(ras, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tin-NN"] + name + '_tinNN.tif')
        #write_ob(array, tinCGAL,  jparams["tin-NN"] + name + '_TINnn.obj')
        
        ras, cdt = execute_cgal_cdt(array, res, origin, jparams["size"])#, target_folder)
        tinflat, raspolys = basic_flattening(ras, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tin-Cnst"] + name + '_tinCnst.tif')    
                
    if jparams["dsm"] == "True":
        array, res, origin, ul_origin = get_las(infile, jparams["size"], gnd_only = False)
        name = Path(infile).stem + '_dsm'
        
        rasLin, tinLin = execute_startin(array, res, origin, jparams["size"], 
                                      method='startin-TINlinear')
        tinflat, raspolys = basic_flattening(rasLin, res, origin, jparams["size"], 
                                              tin = tinLin)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tinLinear"] + name + '_tinLinear.tif')
        #tinLin.write_obj(jparams["tinLinear"] + name + '_TINlinear.obj')
    
        rasLap, tinLap = execute_startin(array, res, origin, jparams["size"], 
                                      method='startin-Laplace')
        tinflat, raspolys = basic_flattening(rasLap, res, origin, jparams["size"], 
                                              tin = tinLap)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tinLaplace"] + name + '_tinLaplace.tif')
        tinLap.write_obj(jparams["tinLaplace"] + name + 'TINlaplace.obj')
    
        pdal_idw(array, res, origin, name, jparams)
        ds = rasterio.open(jparams['pdal-idw'] + name + '_idwPDAL.tif')#.ReadAsArray()
        img = ds.read(1)
        img  = np.flip(img, 0)
        ds = None
        tinflat, raspolys = basic_flattening(img, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["pdal-idw"] + name + '_idwPDAL.tif')
        
        ras = execute_idwquad(array, res, origin, jparams["size"],
                    jparams["start_rk"], jparams["pwr"], jparams["minp"], 
                    jparams["incr_rk"], jparams["method"], jparams["tolerance"], 
                    jparams["maxiter"])
        tinflat, raspolys = basic_flattening(ras, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["quad-idw"] + name + '_idwQUAD.tif')

        ras, tinCGAL = execute_cgal(array, res, origin, jparams["size"])
        tinflat, raspolys = basic_flattening(ras, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tin-NN"] + name + '_tinNN.tif')
        #write_ob(array, tinCGAL,  jparams["tin-NN"] + name + '_TINnn.obj')
        
        ras, cdt = execute_cgal_cdt(array, res, origin, jparams["size"])#, target_folder)
        tinflat, raspolys = basic_flattening(ras, res, origin, jparams["size"], tin = False)
        write_geotiff(raspolys, origin, jparams["size"], jparams["tin-Cnst"] + name + '_tinCnst.tif')
             
if __name__ == "__main__":
    main()
