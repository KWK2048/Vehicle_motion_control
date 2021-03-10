import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22

        self.sample_time = 0.01

    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0


class Bicycle(Bicycle):
    def step(self, v, w):
        # ==================================
        #  Implement kinematic model here
        # ==================================

        w = max(-self.w_max, min(w, self.w_max))
        xc_dot = v * np.cos(self.beta + self.theta)   # 线速度 与 beta ，theta相关
        yc_dot = v * np.sin(self.beta + self.theta)
        theta_dot = v * np.cos(self.beta) * np.tan(self.delta) / self.L
        delta_dot = w
        self.xc += xc_dot * self.sample_time
        self.yc += yc_dot * self.sample_time
        self.delta += delta_dot * self.sample_time
        self.theta += theta_dot * self.sample_time    # beta ，theta与 delta方向盘转角速率 相关；这个速率会影响总体速度
        self.beta = np.arctan(self.lr * np.tan(self.delta) / self.L)
        pass



# 以下是课程作业
sample_time = 0.01
time_end = 20
model = Bicycle()
# solution_model = BicycleSolution()
model.reset()
# set delta directly
model.delta = np.arctan(2 / 10)
# solution_model.delta = np.arctan(2/10)

t_data = np.arange(0, time_end, sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(np.pi, 0)

    # x_solution[i] = solution_model.xc
    # y_solution[i] = solution_model.yc
    # solution_model.step(np.pi, 0)

    model.beta = 0
    # solution_model.beta=0

plt.axis('equal')
plt.plot(x_data, y_data, label='Learner Model')
# plt.plot(x_solution, y_solution,label='Solution Model')
plt.legend()
plt.show()

#################