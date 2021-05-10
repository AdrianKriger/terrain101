**Terrain Simplification.**

Given a Digital Elevation Model (DEM) find the least number of vertices to accurately represent the surface. 

[my_code_hw02.py](https://github.com/AdrianKriger/terrain101/blob/main/hw02_a/my_code_hw02.py) implements a Delaunay-based insertion constrained by a threshold to find the 
most likely candidates ~ we refine rather than decimate.

Example data available [here](https://3d.bk.tudelft.nl/courses/backup/geo1015/2018/hw/02/).
Set parameters with [params.json](https://github.com/AdrianKriger/terrain101/blob/main/hw02_a/params.json). Execute with [geo1015_hw02.py](https://github.com/AdrianKriger/terrain101/blob/main/hw02_a/geo1015_hw02.py)

Input: a raster DEM
Outputs include: an .obj of the triangulation and a .tif of the interpolated surface.

Triangulated surface with 100-, 50- and 25m threshold compared to the original below.

![Alt text](https://github.com/AdrianKriger/terrain101/blob/main/hw02_a/hw02_a_smaller.gif)

Consult [hw02_a.ipynb](https://github.com/AdrianKriger/terrain101/blob/main/hw02_a/hw02_a.ipynb) for a point-by-point comparison of the refined minus the original.

Good to know:
  - Its no efficient.
  - All the output are not necessary. Having the functions to produce them are nice to have through.
