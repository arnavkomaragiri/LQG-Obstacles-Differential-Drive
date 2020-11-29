from src.LQG import *

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

Q = 25 * np.identity(5)
R = np.identity(2)

m = 42 / 2.205
bot_width = 18 / 39.37
bot_length = 18 / 39.37

J = (1 / 12) * m * ((bot_width ** 2) + (bot_length ** 2))

drive = DifferentialDriveModel(m, bot_radius, J, Q, R, 0.00001 * np.identity(5), 0.00001 * np.identity(5), gear_ratio, Kt, Kv, resistance, wheel_radius)
x = np.zeros((5, 1))
u = np.array([[12], [12]])

print(drive.getSimulatedSequence(x, u, 5)[-1])

c = np.array([[-0], [-7], [pi / 2]])
obstacles = [np.array([[-12 + 0.1 * i], [-7], [0]]) for i in range(0, 240)]
print(drive.naiveIsConfigurationValid(x, c, obstacles))
print(drive.isConfigurationValid(x, c, obstacles))
