**Ground Filtering.**

Given an unclassified LiDAR dataset:  
1. Classify ground; and  
2. Create an ascii raster Digital Terrain Model (DTM) through:   
   1. Inverse Distance Weighting (idw); and   
   2. a Delaunay-based linear interpolation (tin).
 
Example data available [here](https://3d.bk.tudelft.nl/courses/backup/geo1015/2019/hw/03/).

To classify we execute a [pdal](https://pdal.io/index.html) pipeline that performs:
  - Poisson sampling - identifies outliers and noise - segments ground and non-ground through a Progressive Morphological Filter (PMF) - and refines the classification with a nearest neighbour consensus.

Set parameters with [params.json](https://github.com/AdrianKriger/terrain101/blob/main/hw03/params.json) 
~ `urban` and `rural` either `True`/`False`; and `idw` and `tin` either `True`/`False`. Execute with [geo1015_hw03.py](https://github.com/AdrianKriger/terrain101/blob/main/hw03/geo1015_hw03.py). 
Output includes a DTM, and .obj of the triangulated surface and as a bonus a DSM.
[fancyPlotting.py](https://github.com/AdrianKriger/terrain101/blob/main/hw03/fancyPlotting.py) to render the ground (as below) via [pyvista](https://docs.pyvista.org/).

![Alt text](https://github.com/AdrianKriger/terrain101/blob/main/hw03/hw03_orbit_2.gif)

Consult [hw03_GroundFilter.ipynb](https://github.com/AdrianKriger/terrain101/blob/main/hw03/hw03_GroundFilter.ipynb) for a look at the result.

Good to know:
  - PMF performs well in the forested (rural) area but incorrectly classifies a few points ~ planar surfaces at strange angles (high pitched roof) ~ in the urban dataset. [covariancefeatures](https://pdal.io/stages/filters.covariancefeatures.html) ~ `Verticality`, `SurfaceVariation`, `Planarity`, etc were unable to isolate the misclassified points. Because the area was flat a `Z` threshold was used to create a ground surface. *This is not the best solution. line 223 in [my_code_hw03.py](https://github.com/AdrianKriger/terrain101/blob/main/hw03/my_code_hw03.py) must change for the script to execute successfully in other areas.* A Deep Learning workflow might be more appropriate. If you happen upon this and know of/have implemented an [open3d-ML](https://github.com/intel-isl/Open3D-ML) pipeline to segment ground and create a terrain surface; please raise and issue and let me know the result.
  - note the hole in the urban DSM and the reason. A comprehensive analysis on modern interpolation methods is available [here](https://3d.bk.tudelft.nl/pdfs/synthesis/2020_ahn3_report.pdf). Its implementation [here](https://github.com/tudelft3d/geo1101.2020.ahn3).
