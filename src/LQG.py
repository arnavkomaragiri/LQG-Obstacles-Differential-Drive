import scipy as sp
from scipy.spatial.transform import Rotation as rot
import numpy as np
from math import *

class StateSpaceModel:
    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = np.zeros(A.shape)
        self.C = C

        S = sp.linalg.solve_continuous_are(a = self.A, b = self.B, q = np.matmul(np.matmul(self.C.T, self.Q[:3, :3]), self.C), r = self.R)
        self.L = np.matmul(np.matmul(np.linalg.inv(self.R), self.B.T), S)
        self.E = np.matmul(np.matmul(np.matmul(np.linalg.inv(R), self.B.T), np.linalg.inv((np.matmul(self.B, self.L) - self.A)).T), np.matmul(self.C.T, self.Q[:3, :3]))

    def getKalmanGain(self, prioriNoise, measurementOperator, measurementNoise):
        self.P = sp.linalg.solve_continuous_are(a = self.A.T, b = measurementOperator.T, q = prioriNoise, r = measurementNoise)
        self.K = np.matmul(np.matmul(self.P, measurementOperator.T), np.linalg.inv(measurementNoise))

    def getStochasticModel(self, prioriNoise, measurementOperator, measurementNoise):
        self.getKalmanGain(prioriNoise, measurementOperator, measurementNoise)

        # print("L: \n", self.L)
        # print("R: \n", np.linalg.inv(self.R))
        # print("B * L: \n", np.linalg.inv((np.matmul(self.B, self.L) - self.A).T))
        # print("A: \n", self.A)
        # print("C^T * Q: \n", np.matmul(self.C.T, self.Q[:3, :3]))
        # print("Magic: \n", np.matmul(np.linalg.inv(R), self.B.T), "\n", np.linalg.inv((np.matmul(self.B, self.L) - self.A)).T)

        a00 = self.A
        a01 = np.matmul(self.K, measurementOperator)
        a10 = -np.matmul(self.B, self.L)
        a11 = self.A + a10 - a01

        b = np.matmul(self.B, self.E)
        # print("B unaugmented: \n", self.B)
        # print("E unaugmented: \n", self.E)

        m00 = prioriNoise
        m11 = np.matmul(np.matmul(self.K, measurementNoise), self.K.T)

        A = np.concatenate(
            (np.concatenate((a00, a01), axis = 1), np.concatenate((a10, a11), axis = 1)),
                axis = 0
        )

        B = np.concatenate((b, b), axis = 0)

        m01 = np.zeros((A.shape[0] - m00.shape[0], A.shape[1] - m00.shape[1])).T

        M = np.concatenate(
            (np.concatenate((m00, m01), axis = 1), np.concatenate((m01.T, m11), axis = 1)),
                axis = 0
        )

        Y = np.concatenate(
            (np.concatenate((self.P, m01), axis = 1), np.concatenate((m01.T, np.zeros(self.P.shape)), axis = 1)),
                axis = 0
        )

        C = np.concatenate((self.C, np.zeros((3, 5))), axis = 1)

        return StochasticClosedFormModel(A, B, C, M, Y)

class Ellipsoid:
    def __init__(self, mean, a, b, c, sortedEigenvectors):
        # print("Basis: ", sortedEigenvectors)
        self.mean = mean
        self.a = a
        self.b = b
        self.c = c

        xyAngle = atan2(sortedEigenvectors[0][0, 1], sortedEigenvectors[0][0, 0]) # Rotation about z
        xzAngle = atan2(sortedEigenvectors[0][0, 2], sortedEigenvectors[0][0, 0]) # Rotation about y
        roll = atan2(sortedEigenvectors[1][0, 1], sortedEigenvectors[1][0, 2]) # Rotation about x

        xRot = np.array([1, 0, 0]) * -roll
        yRot = np.array([0, 1, 0]) * -xzAngle
        zRot = np.array([0, 0, 1]) * -xyAngle

        self.rotationOperator = rot.from_rotvec([
            yRot,
            zRot,
            xRot
        ])

        self.reduceRotation()

    def scale(self, s):
        self.a *= s
        self.b *= s
        self.c *= s

    def isLocalPointInEllipse(self, point):
        newPoint = point - self.mean
        return (1 - (newPoint[0, 0] ** 2) / (self.a ** 2) - (newPoint[0, 1] ** 2) / (self.b ** 2)) >= 0

    def isPointInEllipse(self, point):
        newPoint = point - self.mean
        newPoint = np.matmul(np.matrix(self.rotationOperator.as_matrix()), newPoint)
        return ((1 - (newPoint[0, 0] ** 2) / (self.a ** 2) - (newPoint[1, 0] ** 2) / (self.b ** 2)) >= 0)

    def reduceRotation(self):
        r = np.identity(3)
        for transform in self.rotationOperator.as_matrix():
            r = np.matmul(np.matrix(transform), r)
        self.rotationOperator = rot.from_matrix(r)

    def rotateEllipse(self, rotationMatrix):
        r = np.identity(3)
        for transform in self.rotationOperator:
            r = np.matmul(np.matrix(transform.as_matrix()), r)
        self.rotationOperator = rot.from_matrix(np.matmul(rotationMatrix, r))
        return self

    def rotateEllipseAboutTheta(self, angle, degrees = False):
        mat = np.matrix(rot.from_euler('z', angle, degrees).as_matrix())
        return self.rotateEllipse(mat)

    def expandEllipse(self, a, b, c):
        self.a += a
        self.b += b
        self.c += c

        return self

class StochasticClosedFormModel:
    TIME_HORIZON = 10
    dt = 0.1

    def __init__(self, A, B, C, M, Y):
        def F(t):
            return sp.linalg.expm(A * t)

        def G(t):
            # print("B in G: \n", B)
            return np.matmul(np.matmul(np.linalg.inv(A), F(t) - np.identity(A.shape[0])), B)

        self.F = F
        self.G = G
        self.Ycache = [Y]

        def cov(t):
            F = self.F(t)
            res = np.matmul(np.matmul(F, Y), F.T)

            if t / self.dt >= len(self.Ycache):
                for i in range(len(self.Ycache), int(t / self.dt)):
                    F = self.F(i * self.dt)
                    self.Ycache += [self.Ycache[i - 1] + np.matmul(np.matmul(F, M), F.T)]
                res += self.Ycache[-1]
            else:
                res += self.Ycache[int(t / self.dt)]

            return res

        self.Y = cov
        self.C = C

    def getCovariance(self, t):
        # F = self.F(t)
        # G = self.G(t)
        Y = self.Y(t)

        # print("F: \n", F)
        # print("Y: \n", Y)
        # print("G: \n", G)
        # print("C: \n", self.C)

        # magicMatrix = np.matmul(self.C, G)
        # magicMatrix[-1] += 0.1
        # print(magicMatrix)
        return np.matrix(np.matmul(np.matmul(self.C, Y), self.C.T)) #np.matmul(np.matmul(np.linalg.inv(magicMatrix), np.matmul(np.matmul(self.C, Y), self.C.T)), np.linalg.inv(magicMatrix).T)

    def getEllipsoid(self, c, t):
        cov = self.getCovariance(t)
        # print(cov)
        tmpEigs = np.linalg.eig(cov)
        eigs = []
        for i in range(len(tmpEigs[0])):
            eigs += [[abs(tmpEigs[0][i]), tmpEigs[1][i]]]
        eigs = sorted(eigs, key = lambda x: x[0], reverse = True)
        # print("Eigs: ", eigs)
        # TODO: Use magical table of statistics magic to turn eigenvalues into axes magnitudes
        chi = 11.345 # Note: This corresponds to 1% probability of collision
        return Ellipsoid(c, 2 * sqrt(chi * eigs[0][0]), 2 * sqrt(chi * eigs[1][0]), 2 * sqrt(chi * eigs[2][0]), [eig[1] for eig in eigs])

    def getTransformedObstacle(self, obstacle, x, t):
        augmentedX = np.concatenate((x, x), axis = 1)
        F = self.F(t)
        G = self.G(t)

        return np.matmul(np.linalg.inv(np.matmul(self.C, G)), obstacle - np.matmul(np.matmul(self.C, F), augmentedX))

    def isCandidateTargetValid(self, c, x0, obstacles):
        t = 0

        while (t := t + self.dt) < self.TIME_HORIZON:
            transformedObstacles = [self.getTransformedObstacle(obs, x0, t) for obs in obstacles]
            for obs in transformedObstacles:
                if self.getEllipsoid(c, t).isPointInEllipse(obs):
                    return False

        return True

class DifferentialDriveModel:
    dt = 0.1
    TIME_HORIZON = 40

    def __init__(self, m, r, J, Q, R, modelNoise, measurementNoise, gear_ratio, Kt, Kv, resistance, wheel_radius):
        C = [-((gear_ratio ** 2) * Kt) / (Kv * resistance * (wheel_radius ** 2)),
             (gear_ratio * Kt) / (resistance * r),
             -((gear_ratio ** 2) * Kt) / (Kv * resistance * (wheel_radius ** 2)),
             (gear_ratio * Kt) / (resistance * r)]
        self.m = m
        self.r = r
        self.J = J
        self.C = C
        self.Q = Q
        self.R = R
        self.modelNoise = modelNoise
        self.measurementNoise = measurementNoise

    def linearize(self, x, theta = 0):
        v = (x[3, 0] + x[4, 0]) / 2
        if v == 0:
            v = 0.1
        G1 = ((1 / self.m) + ((self.r ** 2) / self.J))
        G2 = ((1 / self.m) - ((self.r ** 2) / self.J))
        A = np.matrix([[0, 0, -v * sin(theta),       0.5 * cos(theta),     0.5 * cos(theta)],
                       [0, 0,  v * cos(theta),       0.5 * sin(theta),     0.5 * sin(theta)],
                       [0, 0,               0,      -1 / (2 * self.r),     1 / (2 * self.r)],
                       [0, 0,               0,         G1 * self.C[0],        G2 * self.C[2]],
                       [0, 0,               0,         G2 * self.C[0],        G1 * self.C[2]]])

        B = np.matrix([[0,                          0],
                       [0,                          0],
                       [0,                          0],
                       [G1 * self.C[1], G2 * self.C[3]],
                       [G2 * self.C[1], G1 * self.C[3]]])

        C = np.concatenate((np.identity(3), np.zeros((3, 2))), axis = 1)

        return StateSpaceModel(A, B, C, self.Q, self.R)

    def evaluateNLModel(self, x, u):
        v = (x[3, 0] + x[4, 0]) / 2
        G1 = ((1 / self.m) + ((self.r ** 2) / self.J))
        G2 = ((1 / self.m) - ((self.r ** 2) / self.J))

        a = G1 * self.C[0] * x[3, 0] + G2 * self.C[2] * x[4, 0] + G1 * self.C[1] * u[0, 0] + G2 * self.C[3] * u[1, 0]
        b = G2 * self.C[0] * x[3, 0] + G1 * self.C[2] * x[4, 0] + G2 * self.C[1] * u[0, 0] + G1 * self.C[3] * u[1, 0]

        return np.array([[v * cos(x[2, 0])],
                         [v * sin(x[2, 0])],
                         [(x[4, 0] - x[3, 0]) / (2 * self.r)],
                         [a],
                         [b]])

    def getRotation(self, theta):
        return np.matrix([[cos(theta), -sin(theta), 0],
                          [sin(theta),  cos(theta), 0],
                          [         0,           0, 1]])

    def getRotation5d(self, theta):
        return np.matrix([[cos(theta), -sin(theta), 0, 0, 0],
                          [sin(theta),  cos(theta), 0, 0, 0],
                          [         0,           0, 1, 0, 0],
                          [         0,           0, 0, 1, 0],
                          [         0,           0, 0, 0, 1]])


    def rk4Predict(self, x, c, dt):
        model = self.linearize(x, theta = 0)
        c = np.matmul(self.getRotation(-x[2, 0]), c)
        u = -np.matmul(model.L, x) + np.matmul(model.E, c)
        k1 = self.evaluateNLModel(x, u)
        k2 = self.evaluateNLModel(x + 0.5 * dt * k1, u)
        k3 = self.evaluateNLModel(x + 0.5 * dt * k2, u)
        k4 = self.evaluateNLModel(x + 0.5 * dt * k3, u)

        dx = (1 / 6) * dt * (k1 + (2 * k2) + (2 * k3) + k4)
        dTheta = dx[2, 0]
        dx = np.matmul(self.getRotation5d(dTheta), np.matmul(self.getPoseExponential(dTheta), np.matmul(self.getRotation5d(-dTheta), dx)))
        return x + dx

    def getPoseExponential(self, omega):
        if abs(omega) < 0.001:
            return np.identity(5)
        else:
            return np.matrix([[sin(omega) / omega, (cos(omega) - 1) / omega, 0, 0, 0],
                              [(1 - cos(omega)) / omega, sin(omega) / omega, 0, 0, 0],
                              [                       0,                  0, 1, 0, 0],
                              [                       0,                  0, 0, 1, 0],
                              [                       0,                  0, 0, 0, 1]])

    def rk4Simulate(self, x, u, dt):
        k1 = self.evaluateNLModel(x, u)
        k2 = self.evaluateNLModel(x + 0.5 * dt * k1, u)
        k3 = self.evaluateNLModel(x + 0.5 * dt * k2, u)
        k4 = self.evaluateNLModel(x + 0.5 * dt * k3, u)

        dx = (1 / 6) * dt * (k1 + (2 * k2) + (2 * k3) + k4)
        dTheta = dx[2, 0]
        dx = np.matmul(self.getRotation5d(dTheta), np.matmul(self.getPoseExponential(dTheta), np.matmul(self.getRotation5d(-dTheta), dx)))
        return x + dx

    def predict(self, x0, c, t, dt = 0.1):
        x = [x0]
        for i in range(1, int(t / dt) + 1):
            x += [self.rk4Predict(x[i - 1], c, dt)]

        return x[-1]

    def getStateSequence(self, x0, c, t, dt = 0.1):
        x = [x0]
        for i in range(1, int(t / dt) + 1):
            x += [self.rk4Predict(x[i - 1], c, dt)]
            if np.linalg.norm(x[i][:2, 0] - c[:2, 0]) < self.r:
                break
        return x

    def getCovarianceSequence(self, x0, c, t = -1):
        # NOTE: I hate this even more
        # To the professor/industry professional who has spent their entire
        # life writing python code and is now reading this monstrosity,
        # I apologize.
        if t == -1:
            t = self.TIME_HORIZON
        stochasticModel = self.linearize(x0, theta = 0).getStochasticModel(self.modelNoise, np.identity(5), self.measurementNoise)
        return [np.matmul(np.matmul(self.getRotation(x0[2, 0]), stochasticModel.getCovariance(i * self.dt)), self.getRotation(x0[2, 0]).T) for i in range(1, int(t / self.dt))]

    def getEllipseSequence(self, x0, c, t = -1):
        if t == -1:
            t = self.TIME_HORIZON
        x = self.getStateSequence(x0, c, self.TIME_HORIZON)
        stochasticModel = self.linearize(x0, theta = 0).getStochasticModel(self.modelNoise, np.identity(5), self.measurementNoise)
        return [stochasticModel.getEllipsoid(x[i][:3, 0], i * self.dt).rotateEllipseAboutTheta(x0[2, 0]).expandEllipse(self.r, self.r, self.r) for i in range(1, int(t / self.dt))]

    def getPointCloudSequence(self, x0, c, useEllipse = False):
        x = self.getStateSequence(x0, c, self.TIME_HORIZON)
        return zip(x, self.getCovarianceSequence(x0, c, t = int(len(x) * self.dt)))

    def getSimulatedSequence(self, x0, u, t, dt = 0.1):
        x = [x0]
        for i in range(1, int(t / dt) + 1):
            x += [self.rk4Simulate(x[i - 1], u, dt)]

        return x

    def probabilityOfSuccess(self, x0, c, obstacles, pointCloudSequence = []):
        if pointCloudSequence == []:
            pointCloudSequence = self.getPointCloudSequence(x0, c)

        prob = 1.0
        maxProb = 0
        sumProb = 0
        sumItems = 0
        numItems = 0

        # NOTE: I hate this
        k = 1.269522417846096e-05

        for pointCloud in pointCloudSequence:
            covariance = pointCloud[1]

            for obs in obstacles:
                offset = obs - (pointCloud[0][:3])
                exponentiatedMatrix = exp(-0.5 * float(np.matmul(np.matmul(offset.T, np.linalg.inv(covariance)), offset)))
                obstacleCollisionProbability = k * (np.linalg.det(covariance) ** (-1 / 2)) * exponentiatedMatrix

                # maxProb = max(maxProb, obstacleCollisionProbability)
                # sumProb += exponentiatedMatrix
                prob *= (1 - min(1, obstacleCollisionProbability))

            if prob < 0.001:
                return 0

            # if abs(sumProb) > 0.01:
                # item = (sqrt(np.linalg.det(covariance)) / sumProb)
                # print(item)
                # print("Sum Prob: ", sumProb)
                # sumItems += item
                # numItems += 1

        # print(sumItems / numItems)
        return (prob, pointCloudSequence)

    def isConfigurationValid(self, x0, c, obstacles, ellipseSequence = []):
        if ellipseSequence == []:
            ellipseSequence = self.getEllipseSequence(x0, c)

        for ellipse in ellipseSequence:
            for obs in obstacles:
                if ellipse.isPointInEllipse(obs):
                    # print("Obstacle that Failed: \n", obs)
                    # print("Ellipse: ", [ellipse.a, ellipse.b, ellipse.c])
                    return (False, ellipseSequence)

        return (True, ellipseSequence)

    def naiveIsConfigurationValid(self, x0, c, obstacles):
        x = self.getStateSequence(x0, c, self.TIME_HORIZON, dt = self.dt)

        fudge_factor = 7

        for x_t in x:
            # print(x_t)
            for obs in obstacles:
                dist = np.linalg.norm(x_t[:3, 0] - obs)
                # print("Distance From Obstacle at y = {0.2f}: ".format(obs[1, 0]))
                if dist < fudge_factor * self.r:
                    return False

        return True
