from src.LQG import *
from src.Map import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.optimize import BFGS
import time

stall_torque = 0.173 * 120
stall_current = 9.801
voltage = 12
free_speed = (5475.764 / 60) * 2 * pi
free_current = 0.355

resistance = voltage / stall_current
Kv = free_speed / (voltage - free_current * resistance)
Kt = stall_torque / stall_current

gear_ratio = 1 / 3

wheel_radius = 2 / 39.37
bot_radius = 9 / 39.37

Q = np.diag(np.array([20, 20, 20, 1, 1], dtype=float))
R = 0.5 * np.identity(2)

m = 42 / 2.205
bot_width = 18 / 39.37
bot_length = 18 / 39.37

J = (1 / 12) * m * ((bot_width ** 2) + (bot_length ** 2))

drive = DifferentialDriveModel(m, bot_radius, J, Q, R, 0.00001 * np.identity(5), 0.00001 * np.identity(5), gear_ratio, Kt, Kv, resistance, wheel_radius)

x = np.zeros((5, 1))
x[2, 0] = math.pi / 2
obstacles = [np.array([[-12 + 0.1 * i], [-12], [0]]) for i in range(0, 240)]

# res, ellipseSequence = drive.isConfigurationValid(x, c, obstacles)
# prob, pointCloudSequence = drive.probabilityOfSuccess(x, c, obstacles)

# print(res)
# print(prob)

tmpMap = Map().setCurrentPos(x).addLinearObstacle(np.array([[-5], [-5]]), np.array([[5], [-5]]))
obs = tmpMap.getObstacles()
ox, oy = [vector[0, 0] for vector in obs], [vector[1, 0] for vector in obs]
# plt.plot([ox], [oy], marker = 'o', markerSize = 3, color = 'red')
# plt.plot([x[0, 0]], [x[1, 0]], marker = 'o', markerSize = 3, color = 'blue')
# plt.plot([c[0, 0]], [c[1, 0]], marker = 'o',  markerSize = 3, color = 'green')

# px, py = [point[0][0, 0] for point in pointCloudSequence], [point[0][1, 0] for point in pointCloudSequence]
# plt.plot([px], [py], marker = 'o', markerSize = 3, color = 'blue')

# print(tmpMap.isConfigurationValid(drive, c, ellipseSequence = ellipseSequence)[0])
# print(tmpMap.probabilityOfSuccess(drive, c, pointCloudSequence = pointCloudSequence)[0])
# plt.show()

c = np.array([[2], [0], [0], [12], [12]])

x0 = np.zeros((5, 1))
cost = drive.getCostMethod(x0, c, 0.01 * np.identity(5), [])
gradient = drive.getGradientMethod(x0, c, 0.01 * np.identity(5), [])
hess = drive.getHessianMethod(x0, c, 0.01 * np.identity(5), [])
u = np.array([0, 0])
bounds = Bounds([-12, -12], [12, 12])

start = time.time()
# res = minimize(cost, u, method = 'BFGS', jac = gradient, options = {'disp':True})
res = minimize(cost, u, method = 'SLSQP', jac = gradient, hess = hess, bounds = bounds, options = {'ftol': 1e-9, 'disp':True})
end = time.time()

start2 = time.time()
res2 = minimize(cost, u, method = 'trust-constr', jac = gradient, hess = hess, bounds = bounds, options = {'disp':True})
end2 = time.time()

start3 = time.time()
res3 = minimize(cost, u, method = 'Newton-CG', jac = gradient, hess = hess, options = {'xtol':1e-8, 'disp':True})
end3 = time.time()
res3.x = np.clip(res3.x, -12, 12)

print(np.matrix(res3.x).T)
print(drive.simulateCurvilinearModel(np.zeros((5, 1)), res3.x))

dt = 0.1
lookahead_time = 10
trajectoryLength = int(lookahead_time / dt)
print(type(trajectoryLength))
c = np.array([[2], [0], [0]])
controlSequence = np.array([np.array([0, 0]) for _ in range(trajectoryLength)])
traj_start = time.time()
traj, costs = drive.optimizeTrajectory(x0, c, [], controlSequence, 0.01 * np.identity(5))
traj_end = time.time()
print(traj_end - traj_start)
print(traj)

tmp = [np.array([6, 12]) for _ in range(20)]
# traj = drive.initializeTrajectory(x0, tmp, 0.01 * np.identity(5))

traj_x, traj_y = [t[0][0, 0] for t in traj], [t[0][1, 0] for t in traj]
plt.plot(traj_x, traj_y, color = 'red')
plt.show()

print(costs.keys())
print("Please Enter Index of Cost to Analyze")
while input("Would you like to continue: ") != "n":
    i = -1
    while not i in costs:
        i = int(input("Please Enter Index of Cost to Analyze: "))
    cost = costs[i]
    print(traj[i][1])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x = np.arange(-12, 12, 0.25)
    y = np.arange(-12, 12, 0.25)
    x, y = np.meshgrid(x, y)
    positions = np.vstack([x.ravel(), y.ravel()])
    z = np.array([cost(u) for u in positions.T]).reshape(x.shape)
    gz = np.array([np.linalg.norm(gradient(u)) for u in positions.T]).reshape(x.shape)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
