# -*- coding: utf-8 -*-
"""
arkriger ~~

slightly modified code comes from the TU Delft 2020 MSc Geomatics AHN3 improvement. 
https://github.com/khalhoz/geo1101-ahn3-GF-and-Interpolation - under LGPL-3.0 License; and 
https://github.com/tudelft3d/geo1101.2020.ahn3 - under  MIT License 
"""
import math
import json
import numpy as np
import pandas as pd
from laspy.file import File

import pdal

from vector_prepare_local import vector_prepare

def get_las(fpath, size, gnd_only = False):
    """Takes the filepath to an input LAS file and the
    desired output raster cell size. Reads the LAS file, performs basic cleaning and outputs
    the ground points as a numpy array. Also establishes some
    basic raster parameters:
        - the extents
        - the resolution in coordinates
        - the coordinate location of the relative origin (bottom left)
    If called with gnd_only = True, it will ignore non-ground points,
    but this should optimally be done in the PDAL pipeline, not here.
    """
    
    #Import LAS into numpy array 
    array = clean_las(fpath)
    lidar_points = np.array((array['X'], array['Y'], array['Z'])).transpose()
    #Transform to pandas DataFrame
    lidar_df = pd.DataFrame(lidar_points, columns=['x', 'y', 'z'])
    extent = [[lidar_df.x.min(), lidar_df.x.max()],
              [lidar_df.y.min(), lidar_df.y.max()]]
    res = [math.ceil((extent[0][1] - extent[0][0]) / size),
           math.ceil((extent[1][1] - extent[1][0]) / size)]
    origin = [np.mean(extent[0]) - (size / 2) * res[0],
              np.mean(extent[1]) - (size / 2) * res[1]]
    ul_origin = [np.mean(extent[0]) - (size / 2) * res[0],
                 np.mean(extent[1]) + (size / 2) * res[1]]
    
    if gnd_only == True:
        array = array[(array['Classification'] == 2) & (array['Classification'] != 7)]
        #in_np = np.vstack((ground['X'], ground['Y'], ground['Z'])).T
    else:
        # don't .T here. 
        # ~ .T inside the various functions so that pdal_idw does not have to read the file
        array = array[array['Classification'] != 7]
        
    return array, res, origin, ul_origin

def clean_las(fpath):
    """
    pdal pipeline to read .las, crop, identify low noise and identify outliers
    """
    pline={
        "pipeline": [
            {
                "type": "readers.las",
                "filename": fpath
            },
            {
                "type":"filters.crop",
                            #"([xmin, xmax], [ymin, ymax])"
                "bounds":"([-33400, -33200],[-3756000, -3755800])"  # cput
            },
            {
                "type":"filters.elm"
            },
            {
                "type":"filters.outlier",
                "method":"statistical",
                "mean_k": 12,
                "multiplier": 2.2
            },
          ]
        } 
    
    pipeline = pdal.Pipeline(json.dumps(pline))
    #pipeline.validate() 
    pipeline.execute()
    array = pipeline.arrays[0]
    
    return array

def execute_startin(array, res, origin, size, method):
    """Takes the grid parameters and the ground points. Interpolates
    either using the TIN-linear or the Laplace method. Uses a
    -9999 no-data value. Fully based on the startin package.
    """
    import startinpy
    pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    tin = startinpy.DT(); tin.insert(pts)
    ras = np.zeros([res[1], res[0]])
    if method == 'startin-TINlinear':
        def interpolant(x, y): return tin.interpolate_tin_linear(x, y)
    elif method == 'startin-Laplace':
        def interpolant(x, y): return tin.interpolate_laplace(x, y)
    yi = 0
    for y in np.arange(origin[1], origin[1] + res[1] * size, size):
        xi = 0
        for x in np.arange(origin[0], origin[0] + res[0] * size, size):
            tri = tin.locate(x, y)
            if tri != [] and 0 not in tri:
                ras[yi, xi] = interpolant(x, y)
            else: ras[yi, xi] = -9999
            xi += 1
        yi += 1
    return ras, tin

def pdal_idw(array, res, origin, name, jparams):
    """Sets up a PDAL pipeline that reads a ground filtered LAS
    file, and writes it via GDAL. The GDAL writer has interpolation
    options, exposing the radius, power and a fallback kernel width
    to be configured. More about these in the readme on GitHub.
    """
    import pdal
    pline={
        "pipeline": [
            {
                "type":"writers.gdal",
                "filename": jparams['pdal-idw'] + name + '_idwPDAL.tif',
                "resolution": jparams['size'],
                "radius": jparams['pdal-idw-rad'],
                "power": jparams['pdal-idw-pow'],
                "window_size": jparams['pdal-idw-wnd'],
                "output_type": "idw",
                "nodata": -9999,
                "dimension": "Z",
                "origin_x":  origin[0],
                "origin_y":  origin[1],
                "width":  res[0],
                "height": res[1]
            },
          ]
        } 
    
    p = pdal.Pipeline(json.dumps(pline), [array])
    #pipeline.validate() 
    p.execute()    
    
def execute_idwquad(array, res, origin, size,
                    start_rk, pwr, minp, incr_rk, method, tolerance, maxiter):
    """Creates a KD-tree representation of the tile's points and
    executes a quadrant-based IDW algorithm on them. Although the
    KD-tree is based on a C implementation, the rest is coded in
    pure Python (below). Keep in mind that because of this, this
    is inevitably slower than the rest of the algorithms here.
    To optimise performance, one is advised to fine-tune the
    parametrisation, especially tolerance and maxiter.
    More info in the GitHub readme.
    """
    from scipy.spatial import cKDTree
    pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    ras = np.zeros([res[1], res[0]])
    tree = cKDTree(np.array([pts[:,0], pts[:,1]]).transpose())
    yi = 0
    for y in np.arange(origin[1], origin[1] + res[1] * size, size):
        xi = 0
        for x in np.arange(origin[0], origin[0] + res[0] * size, size):
            done, i, rk = False, 0, start_rk
            while done == False:
                if method == "radial":
                    ix = tree.query_ball_point([x, y], rk, tolerance)
                elif method == "k-nearest":
                    ix = tree.query([x, y], rk, tolerance)[1]
                xyp = pts[ix]
                qs = [
                        xyp[(xyp[:,0] < x) & (xyp[:,1] < y)],
                        xyp[(xyp[:,0] > x) & (xyp[:,1] < y)],
                        xyp[(xyp[:,0] < x) & (xyp[:,1] > y)],
                        xyp[(xyp[:,0] > x) & (xyp[:,1] > y)]
                     ]
                if min(qs[0].size, qs[1].size,
                       qs[2].size, qs[3].size) >= minp: done = True
                elif i == maxiter:
                    ras[yi, xi] = -9999; break
                rk += incr_rk; i += 1
            else:
                asum, bsum = 0, 0
                for pt in xyp:
                    dst = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                    u, w = pt[2], 1 / dst ** pwr
                    asum += u * w; bsum += w
                    ras[yi, xi] = asum / bsum
            xi += 1
        yi += 1
    return ras 

def execute_cgal(array, res, origin, size):
    """Performs CGAL-NN on the input points.
    First it removes any potential duplicates from the
    input points, as these would cause issues with the
    dictionary-based attribute mapping.
    Then, it creates CGAL Point_2 object from these points,
    inserts them into a CGAL Delaunay_triangulation_2, and
    performs interpolation using CGAL natural_neighbor_coordinate_2
    by finding the attributes (Z coordinates) via the dictionary
    that was created from the deduplicated points.
    """
    from CGAL.CGAL_Kernel import Point_2
    from CGAL.CGAL_Triangulation_2 import Delaunay_triangulation_2
    from CGAL.CGAL_Interpolation import natural_neighbor_coordinates_2
    pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    s_idx = np.lexsort(pts.T); s_data = pts[s_idx,:]
    mask = np.append([True], np.any(np.diff(s_data[:,:2], axis = 0), 1))
    deduped = s_data[mask]
    cpts = list(map(lambda x: Point_2(*x), deduped[:,:2].tolist()))
    zs = dict(zip([tuple(x) for x in deduped[:,:2]], deduped[:,2]))
    tin = Delaunay_triangulation_2()
    for pt in cpts: tin.insert(pt)
    ras = np.zeros([res[1], res[0]])
    yi = 0
    for y in np.arange(origin[1], origin[1] + res[1] * size, size):
        xi = 0
        for x in np.arange(origin[0], origin[0] + res[0] * size, size):
            nbrs = [];
            qry = natural_neighbor_coordinates_2(tin, Point_2(x, y), nbrs)
            if qry[1] == True:
                z_out = 0
                for nbr in nbrs:
                    z, w = zs[(nbr[0].x(), nbr[0].y())], nbr[1] / qry[0]
                    z_out += z * w
                ras[yi, xi] = z_out
            else: ras[yi, xi] = -9999
            xi += 1
        yi += 1
    return ras, tin

def triangle_area(a, b, c):
    """Computes the area of a triangle with sides a, b, and c.
    Expects an exception to be raised for problematic area
    calculations, in which case it returns False to indicate
    failure.
    """
    try:
        side1 = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
        side2 = np.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)
        side3 = np.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
        sp_pa = (side1 + side2 + side3) * 0.5
        return np.sqrt(sp_pa * (sp_pa - side1) *
                       (sp_pa - side2) * (sp_pa - side3))
    except: return False

def execute_cgal_cdt(array, res, origin, size):#, target_folder):
    """Performs CGAL-CDT on the input points.
    First it removes any potential duplicates from the input points,
    as these would cause issues with the dictionary-based attribute mapping.
    Then, it creates CGAL Point_2 object from these points,
    inserts them into a CGAL Constrained_Delaunay_triangulation_2,
    and then inserts the constraints from shapefiles.
    The constraints do not have elevation values associated.
    Interpolation happens if there are at least two non-constraint
    vertices in a given facet. Otherwise, it either yields the elevation
    of the only non-constraint vertex, or the no-data value is all
    vertices in the facet are constraints.
    It then interpolates (manually, using our code) using TIN-
    linear interpolation via the dictionary-based attribute mapping.
    Extremely long or invalid polygons may mess up the area calculation
    and trigger an exception. These are caught and result in no-data pixels
    which are then filled with values using a median kernel.
    """
    from shapely.geometry import Polygon
    from CGAL.CGAL_Kernel import Point_2
    from CGAL.CGAL_Mesh_2 import Mesh_2_Constrained_Delaunay_triangulation_2
    pts = np.vstack((array['X'], array['Y'], array['Z'])).T
    cdt = Mesh_2_Constrained_Delaunay_triangulation_2()
    s_idx = np.lexsort(pts.T); s_data = pts[s_idx,:]
    mask = np.append([True], np.any(np.diff(s_data[:,:2], axis = 0), 1))
    deduped = s_data[mask]
    cpts = list(map(lambda x: Point_2(*x), deduped[:,:2].tolist()))
    zs = dict(zip([tuple(x) for x in deduped[:,:2]], deduped[:,2]))
    for pt in cpts: cdt.insert(pt)
    poly_fpaths = [
                     './shp/2D_Basic_Building_Footprints_2048Nup.shp',
                     # You can add more resources here.
                  ]
    # wfs_urls =    [
    #                  #('http://3dbag.bk.tudelft.nl/data/wfs', 'BAG3D:pand3d'),
    #                  # You can add more resources here.
    #               ]
    in_vecs = []
    for fpath in poly_fpaths:
        vec = vector_prepare([[origin[0], origin[0] + res[0] * size],
                              [origin[1], origin[1] + res[1] * size]], fpath)
        if len(vec) != 0: in_vecs.append(vec)
    # for wfs in wfs_urls:
    #     vec = wfs_prepare([[origin[0], origin[0] + res[0] * size],
    #                        [origin[1], origin[1] + res[1] * size]],
    #                       wfs[0], wfs[1])
    #     if len(vec) != 0: in_vecs.append(vec)
    def interpolate(pt):
        tr = cdt.locate(Point_2(pt[0], pt[1]))
        v1 = tr.vertex(0).point().x(), tr.vertex(0).point().y()
        v2 = tr.vertex(1).point().x(), tr.vertex(1).point().y()
        v3 = tr.vertex(2).point().x(), tr.vertex(2).point().y()
        vxs = [v1, v2, v3]
        if (pt[0], pt[1]) in vxs:
            try: zs[(pt[0], pt[1])]
            except: return False
        tr_area = triangle_area(v1, v2, v3)
        if tr_area == False: return False
        ws = [triangle_area((pt[0], pt[1]), v2, v3) / tr_area,
              triangle_area((pt[0], pt[1]), v1, v3) / tr_area,
              triangle_area((pt[0], pt[1]), v2, v1) / tr_area]
        try: vx_zs = [zs[vxs[i]] for i in range(3)]
        except: return False
        return vx_zs[0] * ws[0] + vx_zs[1] * ws[1] + vx_zs[2] * ws[2]
    np.seterr(all='raise')
    for polys in in_vecs:
        for poly in polys:
            if len(poly.exterior.coords[:-1]) < 3: continue
            ring, vals, constraints = [], [], []
            for vx in poly.exterior.coords[:-1]:
                val = interpolate(vx)
                if val == False: continue
                ring.append(vx); vals.append(val)
            try:
                Polygon(ring)
                for val in vals: zs[(vx[0], vx[1])] = val
                for vx in ring:
                    constraints.append(cdt.insert(Point_2(vx[0], vx[1])))
                for vx0, vx1 in zip(constraints, np.roll(constraints, -1)):
                    cdt.insert_constraint(vx0, vx1)
            except: continue
            for interior in poly.interiors:
                ring, vals, constraints = [], [], []
                for vx in interior.coords:
                    val = interpolate(vx)
                    if val == False: continue
                try:
                    Polygon(ring)
                    for val in vals: zs[(vx[0], vx[1])] = val
                    for vx in ring:
                        constraints.append(cdt.insert(Point_2(vx[0], vx[1])))
                    for vx0, vx1 in zip(constraints, np.roll(constraints, -1)):
                        cdt.insert_constraint(vx0, vx1)
                except: continue
    ras = np.zeros([res[1], res[0]])
    yi = 0
    for y in np.arange(origin[1], origin[1] + res[1] * size, size):
        xi = 0
        for x in np.arange(origin[0], origin[0] + res[0] * size, size):
            val = interpolate((x, y))
            if val == False: ras[yi, xi] = -9999
            else: ras[yi, xi] = val
            xi += 1
        yi += 1
    np.seterr(all='warn')
    return ras, cdt

def basic_flattening(raster, res, origin, size, tin = False):
    """Reads some pre-determined vector files, tiles them using
    Lisa's code and "burns" them into the output raster. The flat
    elevation of the polygons is estimated by Laplace-interpolating
    at the locations of the polygon vertices. The underlying TIN
    is constructed from the centre points of the raster pixels.
    Rasterisation takes place via rasterio's interface.
    """
    import startinpy
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    transform = (Affine.translation(origin[0], origin[1])
                 * Affine.scale(size, size))
    x0, x1 = origin[0] + size / 2, origin[0] + ((res[0] - 0.5) * size)
    y0, y1 = origin[1] + size / 2, origin[1] + ((res[1] - 0.5) * size)

    poly_fpaths = [
                     './shp/Stormwater_Waterbodies_2048Nup.shp',
                     # You can add more resources here.
                  ]
    # wfs_urls =    [
    #                  #('http://3dbag.bk.tudelft.nl/data/wfs', 'BAG3D:pand3d'),
    #                  # You can add more resources here.
    #               ]
    #raster  = np.flip(raster, 0)
    in_vecs = []
    for fpath in poly_fpaths:
        vec = vector_prepare([[x0, x1], [y0, y1]], fpath)
        if len(vec) != 0: in_vecs.append(vec)
    # for wfs in wfs_urls:
    #     vec = wfs_prepare([[x0, x1], [y0, y1]], wfs[0], wfs[1])
    #     if len(vec) != 0: in_vecs.append(vec)
    if len(in_vecs) == 0: return
    if tin is False:
        xs, ys = np.linspace(x0, x1, res[0]), np.linspace(y0, y1, res[1])
        xg, yg = np.meshgrid(xs, ys); xg = xg.flatten(); yg = yg.flatten()
        cs = np.vstack((xg, yg, raster.flatten())).transpose()
        data = cs[cs[:,2] != -9999]
        tin = startinpy.DT(); tin.insert(data)
    elevations = []
    for polys in in_vecs:
        for poly, i in zip(polys, range(len(polys))):
            els = []
            for vx in poly.exterior.coords:
                try: els += [tin.interpolate_laplace(vx[0], vx[1])]
                except: pass
            for interior in poly.interiors:
                for vx in interior.coords:
                    try: els += [tin.interpolate_laplace(vx[0], vx[1])]
                    except: pass
            elevations.append(np.median(els))
    shapes = []
    for polys in in_vecs:
        shapes += [(p, v) for p, v in zip(polys, elevations)]
    raspolys = rasterize(shapes, raster.shape, -9999, transform = transform)
    #for yi in range(res[1]):
    for yi in range(res[1] - 1, -1, -1):
        for xi in range(res[0]):
            if raspolys[yi, xi] != -9999: raster[yi, xi] = raspolys[yi, xi]
    return tin, raster

def write_asc(res, origin, size, raster, fpath):
    """Writes the interpolated TIN-linear and Laplace rasters
    to disk using the ASC format. The header is based on the
    pre-computed raster parameters.
    """
    #raster  = np.flip(raster, 0)
    with open(fpath, "w") as file_out:
        file_out.write("NCOLS " + str(res[0]) + "\n")
        file_out.write("NROWS " + str(res[1]) + "\n")
        file_out.write("XLLCORNER " + str(origin[0]) + "\n")
        file_out.write("YLLCORNER " + str(origin[1]) + "\n")
        file_out.write("CELLSIZE " + str(size) + "\n")
        file_out.write("NODATA_VALUE " + str(-9999) + "\n")
        for yi in range(res[1] - 1, -1, -1):
            for xi in range(res[0]):
                file_out.write(str(raster[yi, xi]) + " ")
            file_out.write("\n")

def write_geotiff(raster, origin, size, fpath):
    """Writes the interpolated TIN-linear and Laplace rasters
    to disk using the GeoTIFF format. The header is based on
    the raster array and a manual definition of the coordinate
    system and an identity affine transform.
    """
    import rasterio
    from rasterio.transform import Affine
    transform = (Affine.translation(origin[0], origin[1])
                 * Affine.scale(size, size))
    #raster  = np.flip(raster, 0)
    with rasterio.Env():
        with rasterio.open(fpath, 'w', driver = 'GTiff',
                           height = raster.shape[0],
                           width = raster.shape[1],
                           count = 1,
                           dtype = rasterio.float32,
                           #crs='EPSG:2048',
                           crs = '+proj=tmerc +lat_0=0 +lon_0=19 +k=1 +x_0=0 +y_0=0 +axis=enu +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs',
                           transform = transform
                           ) as out_file:
            out_file.write(raster.astype(rasterio.float32), 1)
                        
