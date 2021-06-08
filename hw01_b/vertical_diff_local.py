# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:11:19 2021

@author: arkriger
"""
import os
import rasterio
from rasterio.transform import Affine
import json
import glob
import laspy 
import numpy as np
import math

from code_AHN3_local import clean_las

def read_file(directory):
    src = rasterio.open(directory)
    width = src.width
    height = src.height 
    bound = src.bounds
    raster_array = src.read(1)
    cell_size = ((src.transform * (src.width, 
                                   src.height)) [0] - (src.transform * (0, 0))[0])/width, 
    ((src.transform * (src.width, src.height)) [1] - (src.transform * (0, 0))[1])/height
    return [raster_array, cell_size, bound, width, height]


def read_PC_Data(directory, jparams):
    
    arr = clean_las(directory)
    if jparams['dtm'] == "True":
        array = arr[(arr['Classification'] == 2) & (arr['Classification'] != 7)]
        pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    
    if jparams['dsm'] == "True":
        array = arr[arr['Classification'] != 7]#) & (arr['Classification'] != 9)]
        pts = np.vstack((array['X'], array['Y'], array['Z'])).T
           
    return pts


def calculate_differences(pc_pts, raster_array, raster_cell_size, raster_bbox, 
                          raster_width, raster_height):
    verticle_differences = np.zeros((raster_height, raster_width))

    for col in range(raster_width):
        for row in range(raster_height):
            if raster_array[row][col] == -9999:
                verticle_differences[row][col] = 0
            
    xmin = raster_bbox[0]
    ymin = raster_bbox[1]
    sizex = raster_cell_size[0]
    sizey = raster_cell_size[0]

    for pt in pc_pts:
        colR = int((pt[0]-xmin)/sizex)
        rowR = int((pt[1]-ymin)/sizey)
        if raster_array[rowR][colR] != -9999:
            verticle_differences[rowR][colR] += pt[2]-raster_array[rowR][colR] 
            
    return verticle_differences


def main():
    """
    - params.json ~ 
    """
    jparams = json.load(open('params_local.json'))
    path = os.getcwd()
    vDiff_name = 'vDiff'
    path = os.path.join(path, vDiff_name)
    os.makedirs(path, exist_ok=True)
    for method in jparams['interpolation_methods']:
        # Gets current working directory
        path = vDiff_name #os.getcwd()
        # Joins the folder that we wanted to create
        folder_name = method
        path = os.path.join(path, folder_name) 
        # Creates the folder, and checks if it is created or not.
        os.makedirs(path, exist_ok=True)
    
    if jparams['dtm'] == 'True':
        with open("vDiff/MAE_cput-report_dtm.txt",'w') as ff:

            
            infile = jparams["input-las"]
            name = infile[6:-4]                   
            pc = read_PC_Data(infile, jparams)
    
            for method in jparams['interpolation_methods']:
                raster = method + "/" + name + "_dtm_" + method + ".tif"
                raster_info = read_file(raster)
                differences_values = calculate_differences(pc, raster_info[0], raster_info[1], 
                                                           raster_info[2], raster_info[3], 
                                                           raster_info[4])
                transform = (Affine.translation(raster_info[2][0], 
                                                raster_info[2][3]) * Affine.scale(raster_info[1][0],
                                                                                  raster_info[1][0]))
                with rasterio.Env():
                    with rasterio.open(vDiff_name + '/' + method + "/" + name + "_dtm_" +'vertical_differences_' + method + '.tif', 'w', 
                                        driver = 'GTiff', height = raster_info[4], 
                                        width = raster_info[3], count = 1, 
                                        dtype = differences_values.dtype, 
                                        crs='+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +axis=enu +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 
                                        transform = transform) as dst: dst.write(differences_values, 1)
            
                abs_sum = 0
                for col in range(raster_info[3]):
                    for row in range(raster_info[4]):
                            abs_sum += abs(differences_values[row][col])
                N = (raster_info[3] * raster_info[4]) - np.count_nonzero(differences_values == 0)
                MAE = abs_sum/N
                ff.write("MAE for " + method + "_" + name + "_dtm" + " = " + str(MAE) + "\n")
                print("MAE for " + method + "_" + name + "_dtm" + " = " + str(MAE))
            
    if jparams['dsm'] == 'True':
        with open("vDiff/MAE_cput-report_dsm.txt",'w') as ff:
            
            infile = jparams["input-las"]
            name = infile[6:-4]                   
            pc = read_PC_Data(infile, jparams)
    
            for method in jparams['interpolation_methods']:
                raster = method + "/" + name + "_dsm_" + method + ".tif"
                raster_info = read_file(raster)
                differences_values = calculate_differences(pc, raster_info[0], raster_info[1], 
                                                           raster_info[2], raster_info[3], 
                                                           raster_info[4])
                transform = (Affine.translation(raster_info[2][0], 
                                                raster_info[2][3]) * Affine.scale(raster_info[1][0],
                                                                                  raster_info[1][0]))
                with rasterio.Env():
                    with rasterio.open(vDiff_name + '/' + method + "/" + name + "_dsm_flat_" +'vertical_differences_' + method + '.tif', 'w', 
                                       driver = 'GTiff', height = raster_info[4], 
                                       width = raster_info[3], count = 1, 
                                       dtype = differences_values.dtype, 
                                       crs='+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +axis=enu +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs', 
                                       transform = transform) as dst: dst.write(differences_values, 1)
            
                abs_sum = 0
                for col in range(raster_info[3]):
                    for row in range(raster_info[4]):
                            abs_sum += abs(differences_values[row][col])
                N = (raster_info[3] * raster_info[4]) - np.count_nonzero(differences_values == 0)
                MAE = abs_sum/N
                ff.write("MAE for " + method + "_" + name + "_dsm" + " = " + str(MAE) + "\n")
                print("MAE for " + method + "_" + name + "_dsm" + " = " + str(MAE))
        ff.close()
       
if __name__ == '__main__':
    main()
