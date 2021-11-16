**Major Assignment 2: *Photogrammetric Processing***

Alternate photogrammtric workflow. Parse an `xyz.csv` point-cloud through a python script to:
-	classify ground and non-ground – with a [PDAL](https://pdal.io/) pipeline;
-	TIN interpolation to raster DTM – via [startinpy](https://github.com/hugoledoux/startinpy/) with Laplace interpolation;
-	Raster DSM interpolation – through a home-baked quadrant-based inverse-distance-weighting; and 
-	hillshade and contour creation – with [GDAL](https://gdal.org/). 

GMS260S. Geomatics 2
Cape Peninsula University of Technology
Engineering and the Built Environment
Civil Engineering and Surveying
