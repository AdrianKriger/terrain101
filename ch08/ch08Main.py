# -*- coding: utf-8 -*-
# env/osm3D
"""
Created on Tue Jul  6 14:29:39 2021
@author: arkriger
"""

import json

from ch08Code import assignZ, getosmBld, writegjson, getosmArea, getXYZ, \
    getBldVertices, getAOIVertices, appendCoords, createSgmts, executeDelaunay, pvPlot,\
        writeObj, output_citysjon, createXYZ
    
def main():
    
    createXYZ('./raster/rasterEle.xyz', './raster/3318DC_clip_utm33s.tif')
    ts = assignZ('./data/fp_proj.geojson', './raster/3318DC_clip_utm33s.tif')
    writegjson(ts, './data/fp_proj_z.geojson')
    
    dis, hs = getosmBld('./data/fp_proj_z.geojson')
    buffer, extent = getosmArea('./data/aoi_proj.geojson')
    
    gdf = getXYZ(dis, buffer, "./raster/rasterEle.xyz")
    ac, c = getBldVertices(dis)
    df2 = appendCoords(gdf, ac)
    #df2 = gdf.append(ac, ignore_index=True)
    
    idx = []
    idx = createSgmts(ac, c, gdf, idx)
    
    acoi, ca = getAOIVertices(buffer, './raster/3318DC_clip_utm33s.tif')
        
    idx = createSgmts(acoi, ca, df2, idx)
    #df3 = df2.append(acoi, ignore_index=True)
    df3 = appendCoords(df2, acoi)
    # reduce precision
    #df3["x"] = df3["x"].subtract(836000)
    #df3["y"] = df3["y"].subtract(6230000)
    pts = df3[['x', 'y', 'z']].values

    t = executeDelaunay(hs, df3, idx)
    
     #-- check with a plot
    pvPlot(t, pts, idx, hs)
    
    # replace precision and some other operations
    #df3["x"] = df3["x"].add(836000)
    #df3["y"] = df3["y"].add(6230000)
    #pts = df3[['x', 'y', 'z']].values
    minz = df3['z'].min()
    maxz = df3['z'].max()
    
    writeObj(pts, t, 'wvft_cput3d.obj')
    output_citysjon(extent, minz, maxz, t, pts, 'citjsn_cput3d.json')
    
    upgrade_cjio('citjsn_cput3d.json', 'citjsnV1_cput3d.json')
    
    
  
if __name__ == "__main__":
    main()