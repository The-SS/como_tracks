import math

import matplotlib.pyplot as plt
import numpy as np


class Teleport:
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def update(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        """
        Print the current pose of the robot.
        """
        return f'x: {self.x}, y: {self.y}'


class SingleIntegrator:
    def __init__(self, x=0., y=0., dt=0.1):
        self.x = x
        self.y = y
        self.dt = dt
        self.u = None

    def update(self, u):
        self.u = u
        self.x += self.u[0] * self.dt
        self.y += self.u[1] * self.dt

    def __str__(self):
        """
        Print the current pose of the robot.
        """
        return f'x: {self.x}, y: {self.y}'


class Unicycle:
    def __init__(self, x=0., y=0., theta=0., v=0., dt=0.1):
        """
        Initialize the unicycle object with initial pose (x, y, theta) and velocity (v).
        dt: time step for simulation
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.dt = dt

    def update(self, v, w):
        """
        Update the unicycle pose using the current velocity and angular velocity (w).
        """
        self.v = v
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta += w * self.dt

    def __str__(self):
        """
        Print the current pose of the unicycle.
        """
        return f'x: {self.x}, y: {self.y}, theta: {self.theta}, v: {self.v}'


class SingleTrack:
    def __init__(self, x=0., y=0., heading=0., v=0., lr=0.5, lf=0.5, dt=0.1):
        """
        Initialize the single track dynamics object with:
         - initial pose (x, y, heading)
         - velocity (v)
         - wheelbase (L = lf + lr where lf=(front_wheel-COM), lr=(COM - rear_wheel))
        dt: time step for simulation
        Inputs: acceleration and steering angle
        """
        self.x = x
        self.y = y
        self.heading = heading
        self.v = v
        self.lf, self.lr = lf, lr
        self.wheelbase = lf + lr
        self.body_slip_angle = lambda steer: np.arctan(lr * np.tan(steer) / self.wheelbase)
        self.dt = dt
        self.u = [0, 0]  # (acceleration and steering)

        self.state = lambda px, py, theta, vel: np.array([px, py, theta, vel])
        self.dyn = lambda state, u: \
            np.array(
                [state[3] * np.cos(state[2] + self.body_slip_angle(u[1])),
                 state[3] * np.sin(state[2] + self.body_slip_angle(u[1])),
                 state[3] * np.sin(self.body_slip_angle(u[1])) / self.lr,
                 u[0]])

    def get_state(self):
        return self.state(self.x, self.y, self.heading, self.v)

    def update(self, u):
        """
        Update the single track model using the current acceleration and steering.
        """
        self.u = u
        state = self.get_state()
        next_state = state + self.dyn(state, u) * self.dt
        self.x, self.y, self.heading, self.v = next_state[0], next_state[1], next_state[2], next_state[3]
        return next_state

    def __str__(self):
        """
        Print state.
        """
        state = self.get_state()
        return f'x: {state[0]}, y: {state[1]}, heading: {state[2]}, v: {state[3]}, steering: {self.u[1]}'


def plot_data(t, x, y, heading=None, steering=None, title=None):
    n_plots = 2
    plt_i = 0
    if heading is not None:
        n_plots += 1
    if steering is not None:
        n_plots += 1

    plt_i += 1
    plt.subplot(n_plots, 1, plt_i)
    if title is not None:
        plt.title(title)
    plt.plot(range(t), x)
    plt.ylabel('x')
    plt_i += 1
    plt.subplot(n_plots, 1, plt_i)
    plt.plot(range(t), y)
    plt.ylabel('y')
    if heading is not None:
        plt_i += 1
        plt.subplot(n_plots, 1, plt_i)
        plt.plot(range(t), heading)
        plt.ylabel('theta')
    if steering is not None:
        plt_i += 1
        plt.subplot(n_plots, 1, plt_i)
        plt.plot(range(t), steering)
        plt.ylabel('steering')
    plt.show()


def test_unicycle():
    t = 1000
    x, y, theta = [], [], []
    robot = Unicycle(x=0., y=0., theta=0., v=0., dt=0.1)
    for _ in range(t):
        robot.update(v=1, w=0)
        x.append(robot.x)
        y.append(robot.y)
        theta.append(robot.theta)
    plot_data(t, x, y, heading=theta, steering=None, title="Unicycle: straight")

    x, y, theta = [], [], []
    robot = Unicycle(x=0., y=0., theta=0., v=0., dt=0.1)
    for _ in range(t):
        robot.update(v=1, w=0.3)
        x.append(robot.x)
        y.append(robot.y)
        theta.append(robot.theta)
    plot_data(t, x, y, heading=theta, steering=None, title="Unicycle: circle")


def test_single_track():
    t = 1000
    x, y, theta, steer = [], [], [], []
    robot = SingleTrack(x=0., y=0., heading=0., v=1., lr=0.1, lf=0.1, dt=0.1)
    for _ in range(t):
        robot.update(u=[0, 0])
        x.append(robot.x)
        y.append(robot.y)
        theta.append(robot.heading)
        steer.append(robot.u[1])
    plot_data(t, x, y, heading=theta, steering=steer, title="SingleTrack: straight")

    x, y, theta, steer = [], [], [], []
    robot = SingleTrack(x=0., y=0., heading=0., v=1., lr=0.1, lf=0.1, dt=0.1)
    for _ in range(t):
        robot.update(u=[1, 0])
        x.append(robot.x)
        y.append(robot.y)
        theta.append(robot.heading)
        steer.append(robot.u[1])
    plot_data(t, x, y, heading=theta, steering=steer, title="SingleTrack: straight + accelerate")

    x, y, theta, steer = [], [], [], []
    robot = SingleTrack(x=0., y=0., heading=0., v=1., lr=0.1, lf=0.1, dt=0.01)
    for _ in range(t):
        robot.update(u=[0, 0.3])
        x.append(robot.x)
        y.append(robot.y)
        theta.append(robot.heading)
        steer.append(robot.u[1])
    plot_data(t, x, y, heading=theta, steering=steer, title="SingleTrack: circle")

    x, y, theta, steer = [], [], [], []
    robot = SingleTrack(x=0., y=0., heading=0., v=1., lr=0.1, lf=0.1, dt=0.01)
    for _ in range(t):
        robot.update(u=[1, 0.3])
        x.append(robot.x)
        y.append(robot.y)
        theta.append(robot.heading)
        steer.append(robot.u[1])
    plot_data(t, x, y, heading=theta, steering=steer, title="SingleTrack: circle + accelerate")


def main():
    test_unicycle()
    test_single_track()


if __name__ == "__main__":
    main()

