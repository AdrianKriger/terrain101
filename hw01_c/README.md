**PDAL DTM pipeline.**

Basic PDAL pipeline to transfrom a .las into a DTM. Process includes Cleaning
- [Eliminate low noise](https://pdal.io/stages/filters.elm.html#filters-elm);
- [Outlier removal](https://pdal.io/stages/filters.outlier.html#filters-outlier);
- [Simple Morphological Filter (SMRF)](https://pdal.io/stages/filters.smrf.html#filters-smrf);
- [Removing approximate coplanar features](https://pdal.io/stages/filters.approximatecoplanar.html#filters-approximatecoplanar); and
- [Nearest Neighbor consensus](https://pdal.io/stages/filters.neighborclassifier.html#filters-neighborclassifier)

and Inverse Distance Weighting [gdal.writers](https://pdal.io/stages/writers.gdal.html).
