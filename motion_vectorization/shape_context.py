# Referenced from: 
# https://github.com/creotiv/computer_vision/blob/master/shape_context/shape_context.py

import numpy as np
import cv2
import math
from scipy.spatial.distance import cdist, cosine
from scipy.optimize import linear_sum_assignment
# import random
# import time


# def plot(img, rotate=False):
#     sc = ShapeContext()
#     sampls = 100

#     points1,t1 = sc.get_points_from_img(img,simpleto=sampls)
#     points2,t2 = sc.get_points_from_img(img2,simpleto=sampls)
#     points2 = (np.array(points2)+30).tolist()

#     if rotate:
#         # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
#         theta = np.radians(90)
#         c, s = np.cos(theta), np.sin(theta)
#         R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
#         points2 = np.dot(np.array(points2), R).tolist()

#     P = sc.compute(points1)
#     print P[0]
#     x1 = [p[1] for p in points1]
#     y1 = [p[0] for p in points1]
#     Q = sc.compute(points2)
#     x2 = [p[1] for p in points2]
#     y2 = [p[0] for p in points2]

#     standard_cost,indexes = sc.diff(P,Q)
    
#     lines = []
#     for p,q in indexes:
#         lines.append(((points1[p][1],points1[p][0]), (points2[q][1],points2[q][0])))
    
#     ax = plt.subplot(121)
#     plt.gca().invert_yaxis()
#     plt.plot(x1,y1,'go', x2,y2, 'ro')
    
#     ax = plt.subplot(122)
#     plt.gca().invert_yaxis()
#     plt.plot(x1,y1,'go',x2,y2,'ro')
#     for p1,p2 in lines:   
#         plt.gca().invert_yaxis()
#         plt.plot((p1[0],p2[0]),(p1[1],p2[1]), 'k-')
#     plt.show()
#     print "Cosine diff:", cosine(P.flatten(), Q.flatten())
#     print "Standard diff:", standard_cost


class ShapeContext(object):

    def __init__(self, nbins_r=5, nbins_theta=12, r_inner=0.1250, r_outer=2.0):
        # number of radius zones
        self.nbins_r = nbins_r
        # number of angles zones
        self.nbins_theta = nbins_theta
        # maximum and minimum radius
        self.r_inner = r_inner
        self.r_outer = r_outer

    def _hungarian(self, cost_matrix):
        """
            Here we are solving task of getting similar points from two paths
            based on their cost matrixes. 
            This algorithm has dificulty O(n^3)
            return total modification cost, indexes of matched points
        """
        # print('cost matrix:', cost_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total = cost_matrix[row_ind, col_ind].mean()
        indexes = zip(row_ind.tolist(), col_ind.tolist())
        return total, indexes

    def get_points_from_img(self, image, threshold=50, simpleto=100, radius=2):
        """
            That is not very good algorithm of choosing path points, but it will work for our case.

            Idea of it is just to create grid and choose points that on this grid.
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dst = cv2.Canny(image, threshold, threshold * 3, 3)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        py, px = np.gradient(image)
        # px, py gradients maps shape can be smaller then input image shape
        points = [index for index, val in np.ndenumerate(dst)
                  if val == 255 and index[0] < py.shape[0] and index[1] < py.shape[1]]
        h, w = image.shape

        _radius = radius
        while len(points) > simpleto:
            newpoints = points
            xr = range(0, w, _radius)
            yr = range(0, h, _radius)
            removed = False
            for p in points:
                if p[0] not in yr and p[1] not in xr:
                    newpoints.remove(p)
                    removed = True
                    if len(points) <= simpleto:
                        T = np.zeros((simpleto, 1))
                        for i, (y, x) in enumerate(points):
                            radians = math.atan2(py[y, x], px[y, x])
                            T[i] = radians + 2 * math.pi * (radians < 0)
                        return points, np.asmatrix(T)
            if not removed:
                break
            _radius += 1

        if len(points) > simpleto:
            idxs = np.uint8(np.linspace(0, len(points), simpleto, endpoint=False))
            points = [points[i] for i in idxs]
        if len(points) < simpleto:
            points.extend([[0, 0]] * (simpleto - len(points)))
        T = np.zeros((simpleto, 1))
        for i, (y, x) in enumerate(points):
            radians = math.atan2(py[y, x], px[y, x])
            T[i] = radians + 2 * math.pi * (radians < 0)

        return points, np.asmatrix(T)

    def _cost(self, hi, hj):
        cost = 0
        num_k = self.nbins_theta * self.nbins_r
        # print(hi.shape, hj.shape)
        flag = hi[:num_k] + hj[:num_k]
        add_cost = (hi[:num_k] - hj[:num_k])**2 / flag
        add_cost[flag==0] = 0
        # cost_a = 0.5 * np.sum(add_cost)
        return 0.5 * np.sum(add_cost)
        # for k in range(self.nbins_theta * self.nbins_r):
        #     if (hi[k] + hj[k]):
        #         cost += ((hi[k] - hj[k])**2) / (hi[k] + hj[k])
        # cost_b = 0.5 * cost
        # print(cost_a, cost_b)
        # return cost * 0.5

    def cost_by_paper(self, P, Q, qlength=None):
        p, _ = P.shape
        p2, _ = Q.shape
        d = p2
        if qlength:
            d = qlength
        C = np.zeros((p, p2))

        Q_p2 = np.tile(Q, [p, 1]) / d
        P_p = np.repeat(P, repeats=p2, axis=0) / p
        # Q_p2 = np.repeat(Q, repeats=p, axis=0)
        # P_p = np.tile(P, [p2, 1])
        num_k = self.nbins_theta * self.nbins_r
        flag = Q_p2 + P_p
        add_cost = (Q_p2 - P_p)**2 / flag
        add_cost[flag==0] = 0
        cost = 0.5 * np.sum(add_cost[..., :num_k], axis=1)
        cost = np.reshape(cost, (p, p2))

        # for i in range(p):
        #     for j in range(p2):
        #         C[i, j] = self._cost(Q[j] / d, P[i] / p)

        # print(C - cost)

        # return C
        return cost

    def compute(self, points):
        """
          Here we are computing shape context descriptor
        """
        t_points = len(points)
        # getting euclidian distance
        r_array = cdist(points, points)
        # getting two points with maximum distance to norm angle by them
        # this is needed for rotation invariant feature
        am = r_array.argmax()
        max_points = [am // t_points, am % t_points]
        # normalizing
        r_array_n = r_array / r_array.mean()
        # create log space
        r_bin_edges = np.logspace(np.log10(self.r_inner), np.log10(self.r_outer), self.nbins_r)
        r_array_q = np.zeros((t_points, t_points), dtype=int)
        # summing occurences in different log space intervals
        # logspace = [0.1250, 0.2500, 0.5000, 1.0000, 2.0000]
        # 0    1.3 -> 1 0 -> 2 0 -> 3 0 -> 4 0 -> 5 1
        # 0.43  0     0 1    0 2    1 3    2 4    3 5
        for m in range(self.nbins_r):
            r_array_q += (r_array_n < r_bin_edges[m])

        fz = r_array_q > 0

        # getting angles in radians
        theta_array = cdist(points, points, lambda u, v: math.atan2((v[1] - u[1]), (v[0] - u[0])))
        norm_angle = theta_array[max_points[0], max_points[1]]
        # making angles matrix rotation invariant
        theta_array = (theta_array - norm_angle * (np.ones((t_points, t_points)) - np.identity(t_points)))
        # removing all very small values because of float operation
        theta_array[np.abs(theta_array) < 1e-7] = 0

        # 2Pi shifted because we need angels in [0,2Pi]
        theta_array_2 = theta_array + 2 * math.pi * (theta_array < 0)
        # Simple Quantization
        theta_array_q = (1 + np.floor(theta_array_2 / (2 * math.pi / self.nbins_theta))).astype(int)

        # building point descriptor based on angle and distance
        nbins = self.nbins_theta * self.nbins_r
        descriptor = np.zeros((t_points, nbins))
        for i in range(t_points):
            sn = np.zeros((self.nbins_r, self.nbins_theta))
            for j in range(t_points):
                if (fz[i, j]):
                    sn[r_array_q[i, j] - 1, theta_array_q[i, j] - 1] += 1
            descriptor[i] = sn.reshape(nbins)

        return descriptor

    def cosine_diff(self, P, Q):
        """
            Fast cosine diff.
        """
        P = P.flatten()
        Q = Q.flatten()
        assert len(P) == len(Q), 'number of descriptors should be the same'
        return cosine(P, Q)

    def diff(self, P, Q, qlength=None, idxs=True):
        """
            More precise but not very speed efficient diff.

            if Q is generalized shape context then it compute shape match.

            if Q is r point representative shape contexts and qlength set to 
            the number of points in Q then it compute fast shape match.

        """
        result = None
        C = self.cost_by_paper(P, Q, qlength)
        result = self._hungarian(C)

        if idxs:
            return result
        else:
            return result[0]