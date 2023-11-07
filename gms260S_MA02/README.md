**Major Assignment 2: *Photogrammetric Processing***

Alternate photogrammetric workflow. Parse an `xyz.csv` point-cloud through a python script to:
-	classify ground and non-ground – with a [PDAL](https://pdal.io/) pipeline;
-	TIN to raster DTM – via [startinpy](https://github.com/hugoledoux/startinpy/) with Laplace interpolation;
-	Raster DSM – through a [home-baked quadrant-based inverse-distance-weighting](https://github.com/AdrianKriger/terrain101/blob/main/gms260S_MA02/MA2_Code.py#L181-L228) (IDW); and 
-	hillshade and contour creation – with [GDAL](https://gdal.org/). 



GMS260S. Geomatics 2  
Cape Peninsula University of Technology  
Engineering and the Built Environment  
Civil Engineering and Surveying
