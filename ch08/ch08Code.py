# -*- coding: utf-8 -*-
# env/osm3D
"""
Created on Tue Jul  6 14:10:33 2021

@author: arkriger
"""
import os
from itertools import chain

import numpy as np
import pandas as pd
import geopandas as gpd
#from shapely.geometry import Point, LineString, Polygon, shape, mapping
from shapely.ops import snap
import shapely.geometry as sg
import fiona
import copy
import json
import geojson

from osgeo import gdal, ogr
from rasterstats import zonal_stats, point_query

import pyvista as pv
import triangle as tr

import matplotlib.pyplot as plt

from cjio import cityjson   

def createXYZ(fout, fin):
    """
    read raster and extract a xyz
    """
    xyz = gdal.Translate(fout,
                         fin,
                         format = 'XYZ')
    xyz = None

def assignZ(vfname, rfname):
    """
    assign a height attribute - mean ground - to the osm vector 
    ~ .representative_point() used instead of .centroid
    """
    ts = gpd.read_file(vfname)
    ts['mean'] = pd.DataFrame(
        point_query(
            vectors=ts['geometry'].representative_point(), 
            raster=rfname))
    
    return ts
    
def writegjson(ts, fname):
    """
    read the rasterstats geojson and create new attributes in osm vector
    ~ ground height, relative building height and roof height.
    write the result to file.
    """
    storeyheight = 2.8
    #-- iterate through the list of buildings and create GeoJSON 
    # features rich in attributes
    footprints = {
        "type": "FeatureCollection",
        "features": []
        }
    
    for i, row in ts.iterrows():
        f = {
        "type" : "Feature"
        }
        # at a minimum we only want building:levels tagged
        if 'building:levels' in row.tags:
            f["properties"] = {}
            
            #-- store all OSM attributes and prefix them with osm_          
            f["properties"]["osm_id"] = row.id
            f["properties"]["osm_tags"] = row.tags
            osm_shape = sg.shape(row["geometry"])
                #-- a few buildings are not polygons, rather linestrings. This converts them to polygons
                #-- rare, but if not done it breaks the code later
            if osm_shape.type == 'LineString':
                osm_shape = sg.Polygon(osm_shape)
                #-- and multipolygons must be accounted for
            elif osm_shape.type == 'MultiPolygon':
                #osm_shape = Polygon(osm_shape[0])
                for poly in osm_shape:
                    osm_shape = sg.Polygon(poly)#[0])
                    #-- convert the shapely object to geojson
            f["geometry"] = sg.mapping(osm_shape)
    
            #-- finally calculate the height and store it as an attribute
            f["properties"]['g_height'] = row["mean"]
            f["properties"]['b_height'] = float(row.tags['building:levels']) * storeyheight + 1.3  
            f["properties"]['r_height'] = f["properties"]['b_height'] + row["mean"]
            footprints['features'].append(f)
                
    #-- store the data as GeoJSON
    with open(fname, 'w') as outfile:
        json.dump(footprints, outfile)

def getXYZ(dis, buffer, filen):
    """
    read xyz to gdf
    """
    df = pd.read_csv(filen, 
                 #"./data/rasterEle_holes.xyz", 
                 delimiter = ' ', header=None,
                 names=["x", "y", "z"])
    
    geometry = [sg.Point(xy) for xy in zip(df.x, df.y)]
    #df = df.drop(['Lon', 'Lat'], axis=1)
    gdf = gpd.GeoDataFrame(df, crs="EPSG:32733", geometry=geometry)
    
    _symdiff = gpd.overlay(buffer, dis, how='symmetric_difference')
    _mask = gdf.within(_symdiff.loc[0, 'geometry'])
    gdf = gdf.loc[_mask]
                     
    gdf = gdf[gdf['z'] != 3.402823466385289e+38]
    gdf = gdf.round(2)
    
    return gdf

def getosmBld(filen):
    """
    read osm buildings to gdf, extract the representative_point() for each polygon
    and create a basic xyz_df;
    - reduce the precision of the holes
    """
    dis = gpd.read_file(filen)
    dis.set_crs(epsg=32733, inplace=True, allow_override=True)
     
    # remove duplicate vertices within tolerance 0.2 
    for index, row in dis.iterrows():
        tmp_gdf = dis.copy()
        tmp_gdf['distance'] = tmp_gdf.distance(row['geometry'])
        closest_geom = list(tmp_gdf.sort_values('distance')['geometry'])[1]
        # I took 1 because index 0 would be the row itself
        snapped_geom = snap(row['geometry'], closest_geom, 0.2)
        dis.loc[index, 'geometry'] = snapped_geom
     
    dis.to_file(filen, driver='GeoJSON') 
    dis = dis[dis.osm_id != 904207929] # need to exclude one building
     
    # create a point representing the hole within each building  
    dis['x'] = dis.representative_point().x
    dis['y'] = dis.representative_point().y
    hs = dis[['x', 'y', 'g_height']].copy()#.values.tolist()
    # subtract constant so the values are less
    #hs["x"] = hs["x"].subtract(836000)
    #hs["y"] = hs["y"].subtract(6230000)
    
    return dis, hs

def getosmArea(filen):
    """
    read osm area to gdf and buffer
    - get the extent for the cityjson
    """
    aoi = gpd.read_file(filen)
    buffer = gpd.GeoDataFrame(aoi, geometry = aoi.geometry)
    buffer['geometry'] = aoi.buffer(150, cap_style = 2, join_style = 2)
    
    extent = [aoi.total_bounds[0] - 250, aoi.total_bounds[1] - 250, 
              aoi.total_bounds[2] + 250, aoi.total_bounds[3] + 250]
    
    return buffer, extent

def getBldVertices(dis):
    """
    retrieve vertices from building footprints ~ without duplicates 
    - these vertices already have a z attribute
    """
    all_coords = []
    dps = 2
    segs = {}
    geoms = {}
    
    for ids, row in dis.iterrows():
        oring, z = list(row.geometry.exterior.coords), row['g_height']
        rounded_z = round(z, dps)
        coords_rounded = []
        #po = []
        for x, y in oring:
            rounded_x = round(x, dps)
            rounded_y = round(y, dps)
            coords_rounded.append((rounded_x, rounded_y, rounded_z))
            all_coords.append([rounded_x, rounded_y, rounded_z])
        #oring.pop()
        #for x, y in oring:
            #all_coords.append([rounded_x, rounded_y, rounded_z])
        for i in range(0, len(coords_rounded)-1):
                    x1, y1, z1 = coords_rounded[i]
                    x2, y2, z2 = coords_rounded[i+1]
                    # deduplicate lines which overlap but go in different directions
                    if (x1 < x2):
                        key = (x1, y1, x2, y2)
                    else:
                        if (x1 == x2):
                            if (y1 < y2):
                                key = (x1, y1, x2, y2)
                            else:
                                key = (x2, y2, x1, y1)
                        else:
                            key = (x2, y2, x1, y1)
                    if key not in segs:
                        segs[key] = 1
                    else:
                        segs[key] += 1
    
    c = pd.DataFrame.from_dict(segs, orient="index").reset_index()
    c.rename(columns={'index':'coords'}, inplace=True)
    
    ac = pd.DataFrame(all_coords, columns=['x', 'y', 'z'])
    ac = ac.sort_values(by = 'z', ascending=False)
    ac.drop_duplicates(subset=['x','y'], keep= 'first', inplace=True)
    ac = ac.reset_index(drop=True)
        
    return ac, c

def getAOIVertices(buffer, fname):
    """
    retrieve vertices from aoi ~ without duplicates 
    - these vertices are assigned a z attribute
    """
    aoi_coords = []
    dps = 2
    segs = {}
    
    for ids, row in buffer.iterrows():
        oring = list(row.geometry.exterior.coords)
       
        coords_rounded = []
        po = []
        for x, y in oring:
            [z] = point_query(sg.Point(x, y), raster=fname)
            rounded_x = round(x, dps)
            rounded_y = round(y, dps)
            rounded_z = round(z, dps)
            coords_rounded.append((rounded_x, rounded_y, rounded_z))
            aoi_coords.append([rounded_x, rounded_y, rounded_z])
        #oring.pop()
        #for x, y in oring:
            #all_coords.append([rounded_x, rounded_y, rounded_z])
        for i in range(0, len(coords_rounded)-1):
                    x1, y1, z1 = coords_rounded[i]
                    x2, y2, z2 = coords_rounded[i+1]
                    # deduplicate lines which overlap but go in different directions
                    if (x1 < x2):
                        key = (x1, y1, x2, y2)
                    else:
                        if (x1 == x2):
                            if (y1 < y2):
                                key = (x1, y1, x2, y2)
                            else:
                                key = (x2, y2, x1, y1)
                        else:
                            key = (x2, y2, x1, y1)
                    if key not in segs:
                        segs[key] = 1
                    else:
                        segs[key] += 1
                        
    ca = pd.DataFrame.from_dict(segs, orient="index").reset_index()
    ca.rename(columns={'index':'coords'}, inplace=True)
    
    acoi = pd.DataFrame(aoi_coords, columns=['x', 'y', 'z'])
    #ac = ac.sort_values('z', 
                        #ascending=False).drop_duplicates(subset=['x','y'], keep='last')
    acoi = acoi.sort_values(by = 'z', ascending=False)
    acoi.drop_duplicates(subset=['x','y'], keep= 'first', inplace=True)
    acoi = acoi.reset_index(drop=True)
    
    return acoi, ca

def appendCoords(gdf, ac):
    df2 = gdf.append(ac, ignore_index=True)
    
    return df2

def createSgmts(ac, c, gdf, idx):
    """
    create a segment list for Triangle
    - indices of vertices [from, to]
    """
    
    l = len(gdf) #- 1
    
    for i, row in c.iterrows():
        frx, fry = row.coords[0], row.coords[1]
        tox, toy = row.coords[2], row.coords[3]

        [index_f] = (ac[(ac['x'] == frx) & (ac['y'] == fry)].index.values)
        [index_t] = (ac[(ac['x'] == tox) & (ac['y'] == toy)].index.values)
        idx.append([l + index_f, l + index_t])
    
    return idx

def executeDelaunay(hs, df3, idx):
    """
    perform Triangle ~ constrained Delaunay with concavitities removed
    - return the simplices: indices of vertices that create the triangles
    """      
    holes = hs[['x', 'y']].round(3).values.tolist()
    pts = df3[['x', 'y']].values #, 'z']].values
        
    A = dict(vertices=pts, segments=idx, holes=holes)
    Tr = tr.triangulate(A, 'pVV')  # the VV will print stats in the cmd
    t = Tr.get('triangles').tolist()
    
    # matplotlib for basic plot
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, aspect='equal')
    tr.plot(ax, **Tr)
    plt.show()
      
    return t
    
def pvPlot(t, pv_pts, idx, hs):
    """
    3D plot with PyVista
    """
    l = np.vstack(idx)
    l = l.reshape([-1, 2])
    twos = np.array([[2]] * len(idx))
    lines = np.append(twos, l, axis=1)
    
    trin = pv.PolyData(pv_pts)
    polygon2 = pv.PolyData(pv_pts)
    holes = pv.PolyData()
    # Make sure it has the same points as the mesh being triangulated
    trin.points = pv_pts
    #polygon2.points = pv_pts
    holes = hs[['x', 'y', 'g_height']].values
    
    faces = np.insert(t, 0, np.full((1, len(t)), 3), axis=1)
    trin.faces = faces
    polygon2.lines = lines
    
    p = pv.Plotter(window_size=[750, 450], notebook=False)#, off_screen=True)
    p.add_mesh(trin, show_edges=True, color="blue", opacity=0.2)
    p.add_mesh(polygon2, color="black", opacity=0.3)#, render_points_as_spheres=True)
    p.add_mesh(holes, color="red")
    
    p.set_background('white')
    p.show()
    
def writeObj(pts, dt, obj_filename):
    """
    basic function to produce wavefront.obj
    """
    f_out = open(obj_filename, 'w')
    for p in pts:
        f_out.write("v {:.2f} {:.2f} {:.2f}\n".format(p[0], p[1], p[2]))

    for simplex in dt:
        f_out.write("f {} {} {}\n".format(simplex[0] + 1, simplex[1] + 1, 
                                          simplex[2] + 1))
    f_out.close()
    
def output_citysjon(extent, minz, maxz, T, pts, outfname):
    """
    basic function to produce LoD1 cityjson terrain
    """     
    cm = doVcBndGeom(extent, minz, maxz, T, pts)    
    json_str = json.dumps(cm, indent=2)
    fout = open(outfname, "w")
    fout.write(json_str)  
    
    #up = 'cjio {0} upgrade_version save {1}'.format(outfname,
                                                     #'citjsnV1_cput3d.json')
    #os.system(up)

    #val = 'cjio citjsnV1_cput3d.json validate'
    #os.system(val)

def doVcBndGeom(extent, minz, maxz, T, pts): 
    #-- create the JSON data structure for the City Model
    cm = {}
    cm["type"] = "CityJSON"
    cm["version"] = "0.9"
    cm["CityObjects"] = {}
    cm["vertices"] = []
    #-- Metadata is added manually
    cm["metadata"] = {
    "datasetTitle": "LoD1 terrain model of CPUT (Bellville) campus",
    "datasetReferenceDate": "2021-07-31",
    "geographicLocation": "Cape Town, South Africa",
    "referenceSystem": "urn:ogc:def:crs:EPSG::32733",
    "geographicalExtent": [
        extent[0],
        extent[1],
        minz ,
        extent[1],
        extent[1],
        maxz
      ],
    "datasetPointOfContact": {
        "contactName": "arkriger",
        "linkedin": "www.linkedin.com/in/adrian-kriger",
        "contactType": "private",
        "github": "https://github.com/AdrianKriger/osm_LoD1_3Dbuildings"
        },
    "metadataStandard": "ISO 19115 - Geographic Information - Metadata",
    "metadataStandardVersion": "ISO 19115:2014(E)"
    }
      ##-- do terrain
    add_terrain_v(pts, cm)
    grd = {}
    grd['type'] = 'TINRelief'
    grd['geometry'] = [] #-- a cityobject can have >1 
      #-- the geometry
    g = {} 
    g['type'] = 'CompositeSurface'
    g['lod'] = 1
    allsurfaces = [] #-- list of surfaces
    add_terrain_b(T, allsurfaces)
    g['boundaries'] = allsurfaces
    #g['boundaries'].append(allsurfaces)
      #-- add the geom 
    grd['geometry'].append(g)
      #-- insert the terrain as one new city object
    cm['CityObjects']['terrain01'] = grd

    return cm

def add_terrain_v(pts, cm):
    #cm['vertices'] = pts
    for p in pts:
        cm['vertices'].append([p[0], p[1], p[2]])
    
def add_terrain_b(T, allsurfaces):
    for i in T:
        allsurfaces.append([[i[0], i[1], i[2]]])
    
def upgrade_cjio(infile, outfile):
    """
    upgrade CityJSON
    """   
    cm = cityjson.load(infile)
    cm.upgrade_version("1.0")
    cityjson.save(cm, outfile)