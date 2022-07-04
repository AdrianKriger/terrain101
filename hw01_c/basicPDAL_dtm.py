import math
import json
import numpy as np
import pandas as pd
#from laspy.file import File

import pdal

def get_las(fpath, size):#, gnd_only = False):
    """Takes the filepath to an input LAS file and the
    desired output raster cell size and establishes some
    basic raster parameters:
        - the extents
        - the resolution in coordinates
        - the coordinate location of the relative origin (bottom left)
    """
    
    #Import LAS into numpy array 
    array = las_array(fpath)
    lidar_points = np.array((array['X'], array['Y'], array['Z'])).transpose()
    #Transform to pandas DataFrame
    lidar_df = pd.DataFrame(lidar_points, columns=['x', 'y', 'z'])
    extent = [[lidar_df.x.min(), lidar_df.x.max()],
              [lidar_df.y.min(), lidar_df.y.max()]]
    res = [math.ceil((extent[0][1] - extent[0][0]) / size),
           math.ceil((extent[1][1] - extent[1][0]) / size)]
    origin = [np.mean(extent[0]) - (size / 2) * res[0],
              np.mean(extent[1]) - (size / 2) * res[1]]
    ul_origin = [np.mean(extent[0]) - (size / 2) * res[0],
                 np.mean(extent[1]) + (size / 2) * res[1]]
    
    # if gnd_only == True:
    #     array = array[(array['Classification'] == 2) & (array['Classification'] != 7)]
    #     #in_np = np.vstack((ground['X'], ground['Y'], ground['Z'])).T
    # else:
    #     # don't .T here. 
    #     # ~ .T inside the various functions so that pdal_idw does not have to read the file
    #     array = array[array['Classification'] != 7]
        
    return array, res, origin, ul_origin

def las_array(fpath):
    """
    pdal pipeline to read .las
    """
    pline={
        "pipeline": [
            {
                "type": "readers.las",
                "filename": fpath
            },
          ]
        } 
    
    pipeline = pdal.Pipeline(json.dumps(pline))
    #pipeline.validate() 
    pipeline.execute()
    array = pipeline.arrays[0]
    
    return array

def pdal_idwDTM(array, res, origin, size, outFile):
    """Sets up a PDAL pipeline that reads a basic .las array file, and writes it via GDAL. 
    general processing includes 
        - outlier removal, 
        - take care of 0 classifications, 
        - smrf ground filter,
        - filter out coplanar, and
        - nearest neighbour consensus,
    The GDAL writer has interpolation options, exposing the radius, power and a fallback kernel width.
    """
    pline={
        "pipeline": [
            # {
            #     "type":"filters.assign",
            #     "assignment":"Classification[:]=1"
            # },
                ##-- RuntimeError: filters.smrf: Some NumberOfReturns or ReturnNumber values were 0, but not all. Check that all values in the input file are >= 1.
                ##-- fix with assign 0 to 1 ---https://github.com/PDAL/PDAL/issues/2634
            {
                "type": "filters.ferry",
                "dimensions": "=>ReturnNumber, =>NumberOfReturns" 
            },
            {
                "type": "filters.assign",
                "assignment": "NumberOfReturns[:]=1"
            },
            {
                "type": "filters.assign",
                "assignment": "ReturnNumber[:]=1"
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
                "type":"filters.smrf",
                #"returns": "last",
                "ignore":"Classification[7:7]",
                "window":30,
                "slope":0.2,
                "threshold":0.45,
                "cell":1.0
            },
            {
                "type":"filters.range",
                "limits":"Classification[2:2]"
            },
            {
                "type":"filters.approximatecoplanar",
                "knn":10,
                "thresh1":25,
                "thresh2":6
            },
            {
                "type":"filters.range",
                "limits":"Coplanar[0:0]"
            },
            {
                "type" : "filters.neighborclassifier",
                "domain" : "Classification[2:2]",
                "k" : 10
            },
            {
                "type":"writers.gdal",
                "filename": outFile,
                "resolution": size,
                "radius": 15,
                "power": 1,
                "window_size": 15,
                "output_type": "idw",
                "nodata": -9999,
                "dimension": "Z",
                "origin_x":  origin[0],
                "origin_y":  origin[1],
                "width":  res[0],
                "height": res[1],
                "override_srs": "+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +axis=enu +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
            },
          ]
        } 
    
    p = pdal.Pipeline(json.dumps(pline), [array])
    #pipeline.validate() 
    p.execute()    


def main():
    inFile = "./las/Mamre_Lidar.las"
    outFile = "./basic_idwPDAL/mamre10-m_idwPDAL.tif"
    size = 10

    array, res, origin, ul_origin = get_las(inFile, size)
    pdal_idwDTM(array, res, origin, size, outFile)
    
if __name__ == "__main__":
    main()