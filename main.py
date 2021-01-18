from src.LQG import *
from src.Map import *
import matplotlib.pyplot as plt

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
c = np.array([[-0], [-2], [math.pi / 2]])
obstacles = [np.array([[-12 + 0.1 * i], [-12], [0]]) for i in range(0, 240)]

# res, ellipseSequence = drive.isConfigurationValid(x, c, obstacles)
# prob, pointCloudSequence = drive.probabilityOfSuccess(x, c, obstacles)

# print(res)
# print(prob)

tmpMap = Map().setCurrentPos(x).addLinearObstacle(np.array([[-5], [-5]]), np.array([[5], [-5]]))
obs = tmpMap.getObstacles()
ox, oy = [vector[0, 0] for vector in obs], [vector[1, 0] for vector in obs]
plt.plot([ox], [oy], marker = 'o', markerSize = 3, color = 'red')
plt.plot([x[0, 0]], [x[1, 0]], marker = 'o', markerSize = 3, color = 'blue')
plt.plot([c[0, 0]], [c[1, 0]], marker = 'o',  markerSize = 3, color = 'green')

# px, py = [point[0][0, 0] for point in pointCloudSequence], [point[0][1, 0] for point in pointCloudSequence]
# plt.plot([px], [py], marker = 'o', markerSize = 3, color = 'blue')

# print(tmpMap.isConfigurationValid(drive, c, ellipseSequence = ellipseSequence)[0])
# print(tmpMap.probabilityOfSuccess(drive, c, pointCloudSequence = pointCloudSequence)[0])
# plt.show()

cost = drive.getCostMethod(np.zeros((5, 1)), np.array([[2], [0], [0], [2], [2]]), 0.01 * np.identity(5), [])
gradient = drive.getGradientMethod(np.zeros((5, 1)), np.array([[2], [0], [0], [2], [2]]), 0.01 * np.identity(5), [])
u = np.array([0, 0])
bounds = Bounds([-12, 12], [-12, 12])
res = minimize(cost, u, method = 'trust-constr', jac = gradient, hess = SR1(), bounds = bounds, options = {'verbose':1})
