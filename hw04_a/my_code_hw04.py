#-- my_code_hw04.py
#-- Assignment 04 GEO1015/2018
#-- [arkriger] 
#-- [YOUR STUDENT NUMBER] 
 
"""
code to:
    - harvest .las filter through a pdal pipeline
    - some more filtering with open3d
        - including multiple RANSAC plane detection 
            - and refinement through euclidean clustering 
    - write a .ply of the points associated to a plane.
based on:
    - Yueci Deng: https://github.com/yuecideng/Multiple_Planes_Detection
    - Florent Poux: https://towardsdatascience.com/how-to-automate-3d-point-cloud-segmentation-and-clustering-with-python-343c9039e4f5
    - Kevin Dwyer: https://gist.github.com/dwyerk/10561690
"""
import json
import numpy as np
from scipy.linalg import eigh

import numpy as np
import shapely.geometry as geometry
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay

import pandas as pd
import geopandas as gpd

import open3d as o3d
import pdal

import matplotlib.pyplot as plt

import time
from datetime import timedelta

def fit_plane(pts):
    """
    Fits a plane through a set of points using principal component analysis. 
     
    Input:
        pts:    the points to fit a plane through
        
    Output:
        (n,c)   a tuple with a point p that lies on the plane and the normalised normal vector n of the plane.
    """

    # shift points to mean
    mean = np.mean(pts, axis = 0)  
    pts -= mean
    # compute covariance matrix and eigenvalues and eignevectors
    cov = np.cov(pts, rowvar = False)
    evals, evecs = eigh(cov)
    # find smallest eigenvalue, the corresponging eigenvector is our normal n
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]

    n = evecs[:,-1]
    c = mean

    return c, n

def detect_planes(jparams):
    """
    !!! TO BE COMPLETED !!!
     
    Function that reads a LAS file, performs a region growing plane segmentation, and outputs the result to a PLY file.
     
    Input:
        jparams:  a dictionary with the paramameters:
            input-file:
                        the input LAS file
            output-file:
                        the output PLY file
            thinning-factor:        
                        thinning factor used to thin the input points prior to giving them to the segmentation algorithm. A factor of 1x means no thinning, 2x means remove 1/2 of the points, 10x means remove 9/10 of the points, etc.
            minimal-segment-count:  
                        the minimal number of points in a segment.
            epsilon:                
                        distance threshold to be used during the region growing. It is the distance between a candidate point and the plane of the growing region.
            neighbourhood-type:     
                        specifies the type of neighbourhood search to be used, valid values are `knn` and `radius`.
            k:                      
                        size of the knn neighbourhood searches.
            r:                      
                        radius for he fixed-radius neighbourhood searches.
        
    Output:
        none (but output PLY file written to 'output-file')
    """  
    pass


def clsy_pipe(jparams):
    """
    pdal pipeline to read .las, identify low noise, identify outliers, 
    filter height above ground > 2m only, choose class==1 (non-ground),
    approximatecoplanar and estimaterank
    ~ 
    """
    pline={
        "pipeline": [
            {
                "type": "readers.las",
                "filename": jparams['input-las']
            },
            # {
            #     "type":"filters.sample",
            #     "radius": jparams['thinning-factor']
            # },
            {
                "type":"filters.elm"
            },
            {
                "type":"filters.outlier",
                "method":"statistical",
                "mean_k": 12,
                "multiplier": 2.2
            },
            {
                "type": "filters.hag_nn"
            },
            {
                "type": "filters.range",
                "limits": "HeightAboveGround[2:)"
            },
            {
                "type":"filters.range",
                "limits":"Classification[1:1]"
            },
            {
                "type":"filters.approximatecoplanar",
                "knn":12,
                "thresh1":25,
                "thresh2":6
            },
            {
                "type":"filters.range",
                "limits":"Coplanar[1:1]"
            },
            {
                "type":"filters.estimaterank",
                "knn":12,
                "thresh":0.01
            },
            {
                "type":"filters.range",
                "limits":"Rank[2:2]"
            },
            # {
            #     "type":"writers.las",
            #     "filename": jparams['out-cl01']
            # },
          ]
        } 
    
    pipeline = pdal.Pipeline(json.dumps(pline))
    #pipeline.validate() 
    count = pipeline.execute()
    array = pipeline.arrays[0]
    
    return array

def OwnRanEuclDBSN(pcd, jparams):
    """
    multiple RANSAC plane detection  with Euclidean clustering and refinement 
    - through DBSCAN
    """
    segment_models={}
    segments={}
    #max_plane_idx=20
    rest=pcd
    d_threshold = jparams['epsilon']
    
    min_ratio = 0.1 # (float, optional): The minimum left points ratio to end the Detection. Defaults to 0.05.
    N = len(np.asarray(pcd.points))
    count = 0
    seg = 0
    
    while count < (1 - min_ratio) * N:
        colors = plt.get_cmap("tab20")(seg)
        segment_models[seg], inliers = rest.segment_plane(distance_threshold = jparams['epsilon'],
                                                          ransac_n = jparams['k'],
                                                          num_iterations = 1000)
        segments[seg]=rest.select_by_index(inliers)
        labels = np.array(segments[seg].cluster_dbscan(eps=d_threshold, #*10, 
                                                       min_points=10))
        candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
        best_candidate=int(np.unique(labels)[np.where(candidates==np.max(candidates))[0]])
        print("the best candidate is: ", best_candidate)
        rest = rest.select_by_index(inliers, invert=True)+segments[seg].select_by_index(list(np.where(labels!=best_candidate)[0]))
        segments[seg] = segments[seg].select_by_index(list(np.where(labels==best_candidate)[0]))
        segments[seg].paint_uniform_color(list(colors[:3]))
        print("pass", seg + 1, "/", seg, "done.")
        
        count += len(inliers)
        seg += 1
        
    labels = np.array(rest.cluster_dbscan(eps=d_threshold + 1.5, 
                                          min_points=10))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    return segments, seg, rest

def writePLY(p, OwnRanEuclDBSN_seg, seg, rest, jparams):
    """
    basic function to write .ply 
    - with inverse normalized color [0, 1] mapped to [0, 255]
    """
    
    N = len(np.asarray(p.points))          
    header='\n'.join(["ply", "format ascii 1.0",
                      'comment GEO1015 hw04. author - arkriger',
                      'element vertex {}'.format(N), 
                      'property double x',
                      'property double y',
                      'property double z',
                      'property uchar red',  #start of vertex color
                      'property uchar green',
                      'property uchar blue',
                      'property int segment_id',
                      'end_header'])
        
    l = [] 
    
    OldMin = 0
    OldMax = 1
    NewMin = 0
    NewMax = 255
    OldRange = (OldMax - OldMin)  
    NewRange = (NewMax - NewMin)  
     
    for i in OwnRanEuclDBSN_seg:
        pnts = np.array(OwnRanEuclDBSN_seg[i].points)
        #pnts = np.round(pnts, 2)
        col = np.array(OwnRanEuclDBSN_seg[i].colors)
        for p, c in zip(pnts, col):
            l.append([p[0], p[1], p[2], i + 1, 
                      int((((c[0] - OldMin) * NewRange) / OldRange) + NewMin),
                      int((((c[1] - OldMin) * NewRange) / OldRange) + NewMin),
                      int((((c[2] - OldMin) * NewRange) / OldRange) + NewMin)])
        
    pnts = np.array(rest.points)
    #pnts = np.round(pnts, 2)
    col = np.array(rest.colors)
    for t, y in zip(pnts, col):
        l.append([t[0], t[1], t[2], 0,
                  int((((y[0] - OldMin) * NewRange) / OldRange) + NewMin),
                  int((((y[1] - OldMin) * NewRange) / OldRange) + NewMin),
                  int((((y[2] - OldMin) * NewRange) / OldRange) + NewMin)])
        
    np.savetxt(jparams['output-file'], l, delimiter=' ', 
               fmt = ' '.join(['%1.3f']*3 + ['%i']*4),
               header=header, comments='')
  
def concaveHull(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
       
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull
    
    #meanDis = np.mean(pdist(points))

    #coords = np.array([point.coords[0] for point in points])
    tri = Delaunay(points)
    triangles = points[tri.vertices]
    a = ((triangles[:,0,0] - triangles[:,1,0]) ** 2 + (triangles[:,0,1] - triangles[:,1,1]) ** 2) ** 0.5
    b = ((triangles[:,1,0] - triangles[:,2,0]) ** 2 + (triangles[:,1,1] - triangles[:,2,1]) ** 2) ** 0.5
    c = ((triangles[:,2,0] - triangles[:,0,0]) ** 2 + (triangles[:,2,1] - triangles[:,0,1]) ** 2) ** 0.5
    s = ( a + b + c ) / 2.0
    areas = (s*(s-a)*(s-b)*(s-c)) ** 0.5
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:,(0,1)]
    edge2 = filtered[:,(1,2)]
    edge3 = filtered[:,(2,0)]
    edge_points = np.unique(np.concatenate((edge1,edge2,edge3)), axis = 0).tolist()
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    
    return unary_union(triangles), edge_points
 

def lasToPlanes(jparams):
    """ 
    
    """
    start = time.time()
    array = clsy_pipe(jparams)
    
    #xyz = np.array((array['X'], array['Y'], array['Z'])).T
    
     #-- try with rgb and xyz
    aerial = np.array((array['X'], array['Y'], array['Z'], array['Red'], array['Green'], array['Blue'])).T
     #-- open3d only accepts rgb in the range[0, 1]
    X2 = aerial[:, [3, 4, 5]].astype(np.float64) / 255.0
    xyzrgb = np.concatenate((aerial[:,[0, 1, 2]], X2), axis=1)
    
     #-- open3d data-structure
    pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:,[0, 1, 2]])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:,[3, 4, 5]])

     #-- normal estimation 
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=jparams['r'], 
                                                                           max_nn=jparams['k']), 
                         fast_normal_computation=True)
     #-- outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=12, std_ratio=1.0)
     #-- voxel downsampling
    p = cl.voxel_down_sample(voxel_size = jparams['thinning-factor'])
     #-- RANSAC multiple plane detection with euclidean clustering (DBSCAN)
    OwnRanEuclDBSN_seg, seg, rest = OwnRanEuclDBSN(p, jparams)
     #-- timeit
    end = time.time()
    print('runtime:', str(timedelta(seconds=(end - start))))
    
     #-- plot
    o3d.visualization.draw_geometries([OwnRanEuclDBSN_seg[i] for i in range(seg)] + [rest],
                                      window_name='Open3D', width=750, height=350)
    
    writePLY(p, OwnRanEuclDBSN_seg, seg, rest, jparams)
    
    if jparams['shapes'] == 'True':
        shapes = [] 
        for i in OwnRanEuclDBSN_seg:
            pnts = np.array(OwnRanEuclDBSN_seg[i].points)
            xy = pnts[:,[0, 1]]
            shape, edge_pnts = concaveHull(xy, alpha=0.4)
            shapes.append([shape])
        
        df = pd.DataFrame(shapes, columns = ['geometry'])    
        gdf = gpd.GeoDataFrame(df)  
         #-- plot
        gdf.boundary.plot()
         #-- save
        gdf.to_file(jparams["sh_fname"])
    
    
