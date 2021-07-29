**Plane Detection.**

Given an aerial point cloud; identify planar features (rooves).

Input a .`las`.

Procedures include:  
- [pdal](https://pdal.io/):
   -   [noise](https://pdal.io/stages/filters.elm.html#filters-elm), [outlier](https://pdal.io/stages/filters.outlier.html#filters-outlier), [height-above-ground](https://pdal.io/stages/filters.hag_nn.html#filters-hag-nn) and class filtering (to choose non-ground points > 2m above terrain surface);
   -   [approximatecoplanar](https://pdal.io/stages/filters.approximatecoplanar.html) and [estimaterank](https://pdal.io/stages/filters.estimaterank.html) (to remove trees).
- [open3d](http://www.open3d.org/docs/release/):
  - some more [outlier removal](http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html), [voxel downsampling](http://www.open3d.org/docs/0.8.0/tutorial/Basic/pointcloud.html#voxel-downsampling) and [normal estimation](http://www.open3d.org/docs/0.8.0/tutorial/Basic/pointcloud.html#vertex-normal-estimation);
  - multiple [RANSAC plane detection](http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Plane-segmentation) with [DBSCAN clustering](http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#DBSCAN-clustering).

Output a `.ply` with each point associated to a plane.

Set parameters with a basic [json](https://github.com/AdrianKriger/terrain101/blob/main/hw04_a/params.json).  
[geo1015_hw04.py](https://github.com/AdrianKriger/terrain101/blob/main/hw04_a/geo1015_hw04.py) will excute [my_code_hw04.py](https://github.com/AdrianKriger/terrain101/blob/main/hw04_a/my_code_hw04.py).  
[hw04_PlaneDetection.ipynb](https://github.com/AdrianKriger/terrain101/blob/main/hw04_a/hw04_PlaneDetection.ipynb) for a look.

**some notes:**
 - plane detection is iterative without a mmaximum. The best candidates will be chosen. This might execute slowly with large datasets.
 - I colored the points in the `.ply`. You might not want to do this. Or rather: is there a better way?
