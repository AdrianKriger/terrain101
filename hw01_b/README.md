**AHN3 Interpolation Techniques.**

Given an aerial point cloud; find the best interpolation technique to generate both a Digital Terrain Model (DTM) and Digital Surface Model (DSM).
Procedures include:
- idwPDAL (built-in [pdal writers.gdal](https://pdal.io/stages/writers.gdal.html) Inverse Distance Weighting (idw);
- idwQUAD (home-baked quadrant-based k-nearest neighbor);
- tinLinear (Delaunay-based Linear interpolation);
- tinLaplace (Delaunay-based Laplace interpolation);
- tinNN ([CGAL-Delaunay-based](https://doc.cgal.org/latest/Triangulation_2/index.html) [natural_neighbor interpolation](https://doc.cgal.org/latest/Interpolation/group__PkgInterpolation2NatNeighbor.html));
- tinCnst ([CGAL-Constrained Delaunay](https://doc.cgal.org/latest/Triangulation_2/index.html#title23) with Linear interpolation).

The methods were executed to [refine/improve (re-grid) the Actueel Hoogtebestand Nederland (AHN)](https://github.com/tudelft3d/geo1101.2020.ahn3).

We execute this in a local context ~ LiDAR available upon request from [City of Cape Town](https://www.capetown.gov.za/). geopackage of the area-of-interest available here to reproduce the result. 
Input a classified LiDAR dataset (1 and 2 minumum); which is cropped (line 68 of - see .ipynb to choose an extent) with basic outlier and noise detection.

Set parameters with. Execute with. 
Building outlines were added as contraints for the CGAL-Constrained Delaunay. To test the basic_flattening of waterbodies a wetland in the vicinity was harvested. 
building outlines available [here](https://odp-cctegis.opendata.arcgis.com/datasets/4a542172a2cc430898a5e635d688eee3_86/explore). The Wetland Name	PenTech_feature [wetland](). 
