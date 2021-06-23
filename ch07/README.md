**Runoff modelling.**

Given a Digital Terrian Model (dtm) perform *first-order* runoff modelling; that is:
1) calculate both Single and Multiple flow directions (SFD and MFD); and
2) thier respective accumulation.
3) extract a drainage network and watersheds (basins).
4) Calculate Height Above Nearest Drainage (HAND).

Work follows the [Open Source Geospatial Foundation Research and Education Laboratory at North Carolina State University](https://ncsu-geoforall-lab.github.io/geospatial-simulations-course/)

We execute with [GRASS GIS](https://grass.osgeo.org/). 

location for [example data](https://3d.bk.tudelft.nl/courses/backup/geo1015/2019/hw/02/); set with the [epsg.io](https://epsg.io/32755) PROJ.4 definition `+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs`

#import and set region
`r.in.gdal input=C:\{path}\tasmania.tif output=tasmania_dtm`
`g.region raster=tasmania_dtm@tasmania`
 
Note:
a) [`r.watershed`](https://grass.osgeo.org/grass78/manuals/r.watershed.html) uses A<sup>T</sup> least-cost routing algorithm; sink filling is not recommended.
b) `threshold value` is the number of cells that will be the minimum catchment size. If the resolution of the dem raster is, for example, 10x10 meters (each cell=100 sq. meters), then a threshold of 20,000 (=2,000,000 sq. meters) would create catchments of at least 2 sq. kilometers. - with a 30-m resolution raster if our threshold is 10,000 the catchments will be at least 9 sq.km.
c) MFD is default.

#set r.watershed to produce three ouput. direction, accumulation and basins
`r.watershed elevation=tasmania_dtm@tasmania threshold=10000 accumulation=accum_mfd10k drainage=draindir_mfd10k basin=basin_mfd10k`  

|![alt](https://github.com/AdrianKriger/terrain101/blob/main/ch07/draindir_mfd10.png =200x)|![alt](https://github.com/AdrianKriger/terrain101/blob/main/ch07/accum_mfd10k.png =200x)|![alt](https://github.com/AdrianKriger/terrain101/blob/main/ch07/basin_mfd10k.png =200x)|

![Alt text](https://github.com/AdrianKriger/terrain101/blob/main/ch07/draindir_mfd10.png=250x) ![Alt text](https://github.com/AdrianKriger/terrain101/blob/main/ch07/accum_mfd10k.png =250x) ![Alt text](https://github.com/AdrianKriger/terrain101/blob/main/ch07/basin_mfd10k.png =250x)

#do the same for SFD
`r.watershed -s elevation=tasmania_dtm@tasmania threshold=10000 accumulation=accum_sfd10k drainage=drain_sfd10k basin=basin_sfd10k`
![draindir_sfd10]( =250x) ![basin_sfd10k]( =250x)

#`r.flow` can provide complementary datsets. `r.flow` uses a single flow algorithm i.e. all flow is transported to a single cell downslope. 
`r.flow elevation=tasmania_dtm@tasmania flowline=flowline flowlength=flowLength_30m flowaccumulation=flowAcc_30m`

#you'll notice that although `r.watershed` provides an option to output streams; we did not. We extract streams from the accumulation based on values `> 100`.
`r.mapcalc "streams_mfd = if(abs(accum_mfd10K) > 100, 1, null())"`
#the stream raster usually requires thinning
`r.thin input=streams_mfd@tasmania output=streams_mfd_t`
#and convert to vector
`r.to.vect -s input=streams_mfd_t@tasmania output=streams_mfd_t type=line`


#The HAND model represents the differences in elevation between each grid cell / pixel and the elevations of the flowpath-connected downslope where the flow enters the channel. It is the elevation difference between the cell and the stream where the cell drains. HAND gives a good indication of where inundation will occur.
`r.stream.distance stream_rast=streams_mfd_t direction=draindir_mfd10k elevation=tasmania_dtm method=downstream difference=above_stream`

![above_streams]( =250x)
