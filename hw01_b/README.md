**AHN3 Interpolation Techniques.**

Given an aerial point cloud; find the best interpolation technique to generate both a Digital Terrain Model (DTM) and Digital Surface Model (DSM).
Procedures include:
- idwPDAL (built-in [pdal writers.gdal](https://pdal.io/stages/writers.gdal.html) Inverse Distance Weighting (idw);
- idwQUAD (home-baked quadrant-based k-nearest neighbor);
- tinLinear (Delaunay-based Linear interpolation);
- tinLaplace (Delaunay-based Laplace interpolation);
- tinNN ([CGAL-Delaunay-based](https://doc.cgal.org/latest/Triangulation_2/index.html) [natural_neighbor interpolation](https://doc.cgal.org/latest/Interpolation/group__PkgInterpolation2NatNeighbor.html));
- tinCnst ([CGAL-Constrained Delaunay](https://doc.cgal.org/latest/Triangulation_2/index.html#title23) with Linear interpolation).

The methods were executed to [refine/improve (re-grid) the Actueel Hoogtebestand Nederland (AHN)](https://github.com/tudelft3d/geo1101.2020.ahn3). The report is available [here](https://3d.bk.tudelft.nl/pdfs/synthesis/2020_ahn3_report.pdf) and the code [here](https://github.com/khalhoz/geo1101-ahn3-GF-and-Interpolation)

We execute this in a local context ~ LiDAR available upon request from [City of Cape Town](https://www.capetown.gov.za/). geopackage of the area-of-interest available [here](https://github.com/AdrianKriger/terrain101/tree/main/hw01_b/aoi) to reproduce the result. 
Input a classified LiDAR dataset (1 and 2 minumum); which is cropped ([line 68 of code_AHN3_local.py](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/code_AHN3_local.py#L68) - see [CoCT_cput.ipynb](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/CoCT_cput.ipynb) to choose an extent) with basic outlier and noise detection.

Set parameters [with](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/params_local.json). Execute [with](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/geoAHN3_local.py).  
Building outlines were added as contraints for the [CGAL-Constrained Delaunay](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/code_AHN3_local.py#L248). To test the [basic_flattening of waterbodies](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/code_AHN3_local.py#L355) a wetland in the vicinity was harvested. 
building outlines available [here](https://odp-cctegis.opendata.arcgis.com/datasets/4a542172a2cc430898a5e635d688eee3_86/explore). The PenTech_feature from the stormwater dataset available [here](https://odp-cctegis.opendata.arcgis.com/datasets/74fa0c08ca43494d9b92b1431205bfd7_71/explore). 

[vertical_diff_local.py](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/vertical_diff_local.py) will perfrom a mean-accuray-error calculation (*that is LiDAR minus dtm/dsm*). Consult [CoCT_cput.ipynb](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/CoCT_cput.ipynb) for a look at the result. 
[fancyPlotter.py](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/fancyPlotter.py) to visualize the result (as below)

![Alt text](https://github.com/AdrianKriger/terrain101/blob/main/hw01_b/hw01_b_DtmDsm.gif)

Good to know:
- for larger datasets a multi-processor program, as per the [original](https://github.com/khalhoz/geo1101-ahn3-GF-and-Interpolation), might be more effective;
- building outlines and water features should be harvested via feature/map service. [OWSLib](https://github.com/geopython/OWSLib) might be a good start.
