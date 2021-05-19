#-- main.py
#-- GEO1015.2019--hw03
#-- Ravi Peters <r.y.peters@tudelft.nl>

#------------------------------------------------------------------------------
# DO NOT MODIFY THIS FILE!!!
#------------------------------------------------------------------------------

import json, sys
import math
import numpy as np
from laspy.file import File
from my_code_hw03 import filter_ground

def main():
    """
    - params.json ~ set path to .las and rural / urban: "True"; then idw / tin: "True"
                   ~ and path of output.
    """
    jparams = json.load(open('params.json'))
    
    #Read LAS file
    in_File = File(jparams["input-las"], mode = "r")
    header = in_File.header
    
    # xmin, xmax, ymin, ymax
    extents = [[header.min[0], header.max[0]],
               [header.min[1], header.max[1]]]
    # ncols, nrows
    resolution = [math.ceil((extents[0][1] - extents[0][0]) / jparams["grid-cellsize"]),
           math.ceil((extents[1][1] - extents[1][0]) / jparams["grid-cellsize"])]
    # ll
    origin = [np.mean(extents[0]) - (jparams["grid-cellsize"] / 2) * resolution[0],
              np.mean(extents[1]) - (jparams["grid-cellsize"] / 2) * resolution[1]]
    
    if jparams["rural"] == "True": 
        # ~~ choose idw or tin :"True"
        if jparams["idw"] == "True":
            filter_ground(extents, resolution, origin, jparams, idw=True)
        
        if jparams["tin"] == "True":
            filter_ground(extents, resolution, origin, jparams, tin=True)
            
    if jparams["urban"] == "True": 
        # ~~ choose idw or tin :"True"
        if jparams["idw"] == "True":
            filter_ground(extents, resolution, origin, jparams, urban=True, idw=True)
        
        if jparams["tin"] == "True":
            filter_ground(extents, resolution, origin, jparams, urban=True, tin=True)
        
if __name__ == "__main__":
    main()