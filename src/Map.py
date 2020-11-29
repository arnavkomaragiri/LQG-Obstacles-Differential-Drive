import numpy as np
import scipy as sp
import math
import random

class Map:
    dTheta = 0.01

    def __init__(self, obstacles = [], max_dim = 10):
        self.obstacles = obstacles
        self.max_dim = max_dim

    def addObstacle(self, obstacle):
        return Map(obstacles = self.obstacles + [obstacle], max_dim = self.max_dim)

    def addCircularObstacle(self, center, radius):
        theta = 0
        obstacles = []

        while theta < 2 * math.pi:
            obstacles += [center + np.array([radius * math.cos(theta), radius * math.sin(theta)]).T]
            theta += self.dTheta

        return Map(obstacles = self.obstacles + obstacles, max_dim = self.max_dim)

    def addCircularObstacles(self, centers, radii):
        res = self

        if len(centers) != len(radii):
            raise ValueError("Centers list and radius list must be equal lengths")

        for i in range(len(centers)):
            res = res.addCircularObstacle(centers[i], radii[i])

        return res

    def addRandomCircularObstacle(self, max_radius):
        theta = random.uniform(0, 2 * math.pi)
        dist = random.uniform(0, self.max_dim)
        radius = random.uniform(0, max_radius)

        return self.addCircularObstacle(np.array([dist * math.cos(theta), dist * math.sin(theta)]).T, radius)

    def addRandomCircularObstacles(self, n, max_radius, allowIntersectingObstacles = True):
        res = self
        i = 0

        while i < n:
            tmp = res.addRandomCircularObstacle(max_radius)

            if allowIntersectingObstacles:
                res = tmp
                n += 1
            else:
                seen = set()
                for obstacle in tmp.obstacles:
                    if obstacle in seen:
                        break
                    else:
                        seen.add(obstacle)

                if len(seen) == len(tmp.obstacles):
                    res = tmp
                    n += 1
        return res
