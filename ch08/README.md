**Conversion between terrain representations.**

Transform a raster Digital Terrain Model (dtm) to a Triangulated Irregular Network (tin).  

To add a level of complexity;  
    a) Given a set of vector polygons (building footprints and an area of interest) - from [OpenStreetMap](https://wiki.osmfoundation.org/wiki/Main_Page) - perform a  
       Constrained Delaunay triangulation where the buildings are removed from the terrain;  
    b) Do no reduce the quality of the raster (don't cut holes in the dtm); assign a height to the vector - add value to the osm data through new attributes: namely 
       height of ground;  
    c) Output the result to:  
&nbsp;&nbsp;&nbsp;&nbsp;- wavefront.obj; and  
&nbsp;&nbsp;&nbsp;&nbsp;- Level of Detail 1 (LoD1) 3D City Model (cityjson) *[this does not execute correctly]*.
                      
 [ch08Main](https://github.com/AdrianKriger/terrain101/blob/main/ch08/ch08Main.py) will execute [ch08Code](https://github.com/AdrianKriger/terrain101/blob/main/ch08/ch08Code.py). 
 
 *for reproducibility:*
 - A portion of LO19_050M_3318DC is available (original [here](http://www.ngi.gov.za/index.php/online-shop/what-is-itis-portal));
 - osm vector via [these queries](https://github.com/AdrianKriger/osm_LoD1_3Dbuildings/blob/main/osm_lod1_3dbuildingmodel_cput.ipynb);
 - both reprojected to `espg:32733`. [gdal.vectorTranslate and gdal.Warp](https://gdal.org/python/osgeo.gdal-module.html#VectorTranslateOptions) comes in handy.

Constrained Delaunay via [python bindings](https://rufat.be/triangle/) of Shewchuck's; [Triangle](http://www.cs.cmu.edu/~quake/triangle.html).  
builtin [pvPlot](https://github.com/AdrianKriger/terrain101/blob/main/ch08/ch08Code.py#L299) to visualize via [PyVista](https://docs.pyvista.org/) - after the triangulation but before c).
