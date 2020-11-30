import numpy as np
import scipy as sp
import math
import random

class Map:
    dT = 0.01

    def __init__(self, obstacles = [], max_dim = 10):
        self.obstacles = obstacles
        self.max_dim = max_dim

    def addLinearObstacle(self, p1, p2):
        return Map(obstacles = self.obstacles + [[p1, p2, 1]], max_dim = self.max_dim)

    def addCircularObstacle(self, center, radius):
        return Map(obstacles = self.obstacles + [[center, radius, 0]], max_dim = self.max_dim)

    def addLinearObstacles(self, p1List, p2List):
        res = self

        if len(p1List) != len(p2List):
            raise ValueError("P1 and P2 Lists must be equal lengths")

        for i in range(len(p1List)):
            res = res.addLinearObstacle(p1List[i], p2List[i])

        return res

    def addCircularObstacles(self, centers, radii):
        res = self

        if len(centers) != len(radii):
            raise ValueError("Centers list and radius list must be equal lengths")

        for i in range(len(centers)):
            res = res.addCircularObstacle(centers[i], radii[i])

        return res

    def addRandomLinearObstacle(self):
        dist1 = random.uniform(0, self.max_dim)
        dist2 = random.uniform(0, self.max_dim)

        theta1 = random.uniform(0, 2 * math.pi)
        theta2 = random.uniform(0, 2 * math.pi)

        return self.addLinearObstacle(np.array([[dist1 * math.cos(theta1)], [dist1 * math.sin(theta1)]]),
                                        np.array([[dist2 * math.cos(theta2)], [dist2 * math.sin(theta2)]]))

    def addRandomCircularObstacle(self, max_radius):
        theta = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, self.max_dim)
        radius = random.uniform(0, max_radius)

        return self.addCircularObstacle(np.array([[dist * math.cos(theta)], [dist * math.sin(theta)]]), radius)

    def addRandomLinearObstacles(self, n, allowIntersectingObstacles = True):
        res = self
        i = 0

        while i < n:
            tmp = res.addRandomLinearObstacle()

            if allowIntersectingObstacles:
                res = tmp
                i += 1
            else:
                # Helper methods for line-line intersection taken from
                # https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

                def onSegment(p, q, r):
                    if ((q[0, 0] <= max(p[0, 0], r[0, 0])) and (q[0, 0] >= min(p[0, 0], r[0, 0]))) and ((q[1, 0] <= max(p[1, 0], r[1, 0])) and (q[1, 0] >= min(p[1, 0], r[1, 0]))):
                        return True
                    return False

                def orientation(p, q, r):
                    val = ((q[1, 0] - p[1, 0]) * (r[0, 0] - q[0, 0])) - ((q[0, 0] - p[0, 0]) * (r[1, 0] - q[1, 0]))

                    if val > 0:
                        return 1
                    elif val < 0:
                        return 2
                    else:
                        return 0

                def intersect(p1, q1, p2, q2):
                    o1 = orientation(p1, q1, p2)
                    o2 = orientation(p1, q1, q2)
                    o3 = orientation(p2, q2, p1)
                    o4 = orientation(p2, q2, q1)

                    if (o1 != o2) and (o3 != o4):
                        return True

                    if (o1 == 0 and onSegment(p1, p2, q1)) or (o3 == 0 and onSegment(p2, p1, q2)) or (o4 == 0 and onSegment(p2, q1, q2)):
                        return True
                    return False

                failed = False
                for obs in tmp.obstacles[:-1]:
                    if obs[2] == 1 and tmp.obstacles[-1][2] == 1:
                        if intersect(obs[0], obs[1], tmp.obstacles[-1][0], tmp.obstacles[-1][1]):
                            failed = True
                            break

                if not failed:
                    res = tmp
                    i += 1
        return res


    def addRandomCircularObstacles(self, n, max_radius, allowIntersectingObstacles = True):
        res = self
        i = 0

        while i < n:
            tmp = res.addRandomCircularObstacle(max_radius)

            if allowIntersectingObstacles:
                res = tmp
                i += 1
            else:
                failed = False
                for obs in tmp.obstacles[:-1]:
                    # TODO: Implement checks for linear/circular obstacle intersections
                    if obs[2] == 0 and tmp.obstacles[:-1][2] == 0:
                        if np.linalg.norm(obs[0] - tmp.obstacles[-1][0]) <= obs[1] + tmp.obstacles[-1][1]:
                            failed = True
                            break

                if not failed:
                    res = tmp
                    i += 1

        return res

    def getObstacles(self):
        res = []

        for obs in self.obstacles:
            if obs[2] == 0:
                theta = 0
                while theta <= 2 * math.pi:
                    res += [obs[0] + np.array([[obs[1] * math.cos(theta)], [obs[1] * math.sin(theta)]])]
                    theta += self.dT
            elif obs[2] == 1:
                base = 0
                vec = obs[1] - obs[0]
                while base <= 1:
                    res += [obs[0] + base * vec]
                    base += self.dT

        return res
