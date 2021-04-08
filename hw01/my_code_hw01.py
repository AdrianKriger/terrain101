#-- my_code_hw01.py
#-- GEO1015.20019--hw01
#-- [YOUR NAME]
#-- [YOUR STUDENT NUMBER] 
#-- [YOUR NAME]
#-- [YOUR STUDENT NUMBER] 


"""
Simple structured Delaunay triangulation in 2D with Bowyer-Watson algorithm.
Mostly written by Jose M. Espadero ( http://github.com/jmespadero/pyDelaunay2D )
Based on code from Ayron Catteau. Published at http://github.com/ayron/delaunay
Just pretend to be simple and didactic. The only requisite is numpy.
Robust checks disabled by default. May not work in degenerate set of points.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can add any new functions to this unit, but do not import new units.

You can add new methods to the DT class, but the functions that already exist
must have the same name/structure/input/output.

You need to complete the 2 functions:
  1. insert_one_point(self, x, y)
  2. get_voronoi_edges(self)

The data structure that must be used is:
    pt = [x, y]
    r = [pt1, pt2, pt3, neighbour1, neighbour2, neighbour3]  
"""
###### ~~~~~~~~~~~~~~~ ###########~~~~~~~~~~~~~###########
# import module
import numpy as np
from math import sqrt
from itertools import chain

class DT:
    def __init__(self, center=(0, 0), radius=10000):
        self.pts = []
        self.trs = []
        #- create infinite triangle
        #- create 3 vertices
        #self.pts.append([-10000, -10000])
        #self.pts.append([10000, -10000])
        #self.pts.append([0, 10000])
        self.pts.append([-10000, -10000])
        self.pts.append([10000, -10000])
        self.pts.append([10000, 10000])
        self.pts.append([-10000, 10000])
        #- create one triangle
        #self.trs.append([0, 1, 2, -1, -1, -1])
        ### ~~~ ###
        center = np.mean(self.pts, axis=0)
        #print("Center:", center)
        center = np.asarray(center)

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        #T1 = (0, 1, 2)
        #T2 = (2, 3, 1)
        #self.triangles[T1] = [None, None, None]
        #self.triangles[T2] = [T1, None, None]
        
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)
            
        #self.vorcoors = []
        #self.regions = {}
        
    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.pts[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)
    
    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius
        
    def number_of_points(self):
        return len(self.pts)

    def number_of_triangles(self):
        return len(self.triangles)

    def get_delaunay_vertices(self):
        return self.pts

    def get_delaunay_edges(self):
        edges = []
        for tr in self.triangles:
            a = self.pts[tr[0]]
            b = self.pts[tr[1]]
            c = self.pts[tr[2]]
            edges.append(a)
            edges.append(b)
            edges.append(a)
            edges.append(c)
            edges.append(b)
            edges.append(c)
        return edges

    def get_voronoi_edges(self):
        """
        !!! TO BE COMPLETED AND MODIFIED !!!

        The returned list contains only points ([x,y]), the first 2 are one edge, 
        the following 2 are another edge, and so on.

        Thus for 2 edges, one betwen a and b, and the other between c and d:
        edges = [ [x1, y1], [x2, y2], [x43, y43], [x41, y41]]
        """
        ## ~~~~ ###
        vc, vr = self.exportVoronoiRegions()
        #edge = []
        #edges = []
        # for r in vr:
        #     edge = [vc[i] for i in vr[r]]
        #     edge = np.concatenate(edge).tolist()
        #     edge = list(np.repeat(edge, 2))
        #     edge.insert(len(edge),edge.pop(0))
        #     edge = [edge[i:i+2] for i in range(0, len(edge), 2)]
        #     edges.append(edge)
        # edges = list(chain.from_iterable(edges))
        #return edges
        
        #### ~~~~ try polygon
        polygon = []
        for r in vr:
            edge = [vc[i] for i in vr[r]]
            edge = np.concatenate(edge).tolist()
            polygon.append(edge)

        #-- this is a dummy example that shows 2 lines in the area
        # edges.append([100, 100])
        # edges.append([200, 300])
        # edges.append([200, 300])
        # edges.append([450, 450])
        return polygon

    def insert_one_point(self, x, y):
        """
        !!! TO BE COMPLETED !!!
        """
        #self.pts.append((x, y))
        # code added here to create the (Delaunay) triangles
        ### ~~~~ ###
        p = np.asarray([[x, y]])
        idx = len(self.pts)
        # print("coords[", idx,"] ->",p)
        self.pts.append((x, y))

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous
    
        
    def exportExtendedDT(self):
            """Export the Extended Delaunay Triangulation (with the frame vertex).
            """
            return self.coords, self.triangles
        
    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        # Remember to compute circumcircles if not done before
        #for t in self.triangles:
            #self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles]# if a > 3 and b > 3 and c > 3]
        
    def exportVoronoiRegions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data.
        """
        # Remember to compute circumcircles if not done before
        for t in self.triangles:
             self.circles[t] = self.circumcenter(t)
        useVertex = {i: [] for i in range(len(self.pts))}
        vor_coors = []
        index = {}
        # Build a list of coordinates and one index per triangle/region
        for tidx, (a, b, c) in enumerate(sorted(self.triangles)):
            vor_coors.append(self.circles[(a, b, c)][0])
            # Insert triangle, rotating it so the key is the "last" vertex
            useVertex[a] += [(b, c, a)]
            useVertex[b] += [(c, a, b)]
            useVertex[c] += [(a, b, c)]
            # Set tidx as the index to use with this triangle
            index[(a, b, c)] = tidx
            index[(c, a, b)] = tidx
            index[(b, c, a)] = tidx
            #self.vorcoors.append(vor_coors)

        # init regions per coordinate dictionary
        regions = {}
        # Sort each region in a coherent order, and substitude each triangle
        # by its index
        for i in range(4, len(self.pts)):
            v = useVertex[i][0][0]  # Get a vertex of a triangle
            r = []
            for k in range(len(useVertex[i])):
                # Search the triangle beginning with vertex v
                t = [t for t in useVertex[i] if t[0] == v][0]
                r.append(index[t])  # Add the index of this triangle to region
                v = t[1]            # Choose the next vertex to search
            regions[i-4] = r        # Store region.
            #self.regions[i-4] = r
            
        #self.vorcoors.append(vor_coors)
        #self.regions.append(regions)
        return vor_coors, regions
        
