from src.LQG import *
from src.Map import *
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

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

Q = np.diag(np.array([2, 2, 2, 1, 1]))
R = 0.5 * np.identity(2)

m = 42 / 2.205
bot_width = 18 / 39.37
bot_length = 18 / 39.37

J = (1 / 12) * m * ((bot_width ** 2) + (bot_length ** 2))

drive = DifferentialDriveModel(m, bot_radius, J, Q, R, 0.00001 * np.identity(5), 0.00001 * np.identity(5), gear_ratio, Kt, Kv, resistance, wheel_radius)

x = np.zeros((5, 1))
x[2, 0] = math.pi / 2
c = np.array([[2], [0], [0], [2], [2]])
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

x0 = np.zeros((5, 1))
cost = drive.getCostMethod(x0, c, 0.01 * np.identity(5), [])
throwaway_cost = lambda x: (c - x).T @ Q @ (c - x)
gradient = drive.getGradientMethod(x0, c, 0.01 * np.identity(5), [])
u = np.array([12, 12])
bounds = Bounds([-12, 12], [-12, 12])
res = minimize(cost, u, method = 'trust-constr', jac = gradient, hess = SR1(), bounds = bounds, options = {'verbose':3})

print(res.x)
print(drive.simulateCurvilinearModel(np.zeros((5, 1)), res.x))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
x = np.arange(-12, 12, 0.1)
y = np.arange(-12, 12, 0.1)
x, y = np.meshgrid(x, y)
positions = np.vstack([x.ravel(), y.ravel()])
z = np.array([cost(u) for u in positions.T]).reshape(x.shape)
gz = np.array([np.linalg.norm(gradient(u)) for u in positions.T]).reshape(x.shape)
throwaway_z = np.array([throwaway_cost(drive.simulateCurvilinearModel(x0, i.T)) for i in positions.T]).reshape(x.shape)
surf = ax.plot_surface(x, y, gz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# surf2 = ax.plot_surface(x, y, throwaway_z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
