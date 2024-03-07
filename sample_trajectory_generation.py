import os
import math
import numpy as np
import matplotlib.pyplot as plt
from dynamics import *


def transform_point_to_local_frame(point_pos_global: np.ndarray,
                                   robot_pos_global: np.ndarray, robot_heading_global: float):
    """
    Transforms a given point in the global frame to the local robot frame
    :param point_pos_global: x,y position of the point in the global frame (meters)
    :param robot_pos_global: x,y position of the robot in the global frame (meters)
    :param robot_heading_global: heading angle of the robot (rad) (angle between the robot's and world frame's x-axes)
    :return: x,y position of the point in the robot's local coordinate frame
    """
    # calculate the translation vector from the robot origin to the given point
    translation = point_pos_global - robot_pos_global
    # calculate the rotation matrix that transforms from the global world frame to the local robot frame
    rotation = np.array([[np.cos(robot_heading_global), np.sin(robot_heading_global)],
                         [-np.sin(robot_heading_global), np.cos(robot_heading_global)]])
    # calculate the transformed position in the robot frame
    local_pos = np.dot(rotation, translation)
    return local_pos


def find_closest_index(x_arr, y_arr, pos, L):
    closest_dist = float('inf')
    closest_idx = -1
    for i in range(len(x_arr)):
        dist = np.sqrt((x_arr[i]-pos[0])**2 + (y_arr[i]-pos[1])**2)
        if L == dist:
            return i, dist
        elif L < dist < closest_dist:
            closest_dist = dist
            closest_idx = i
    return closest_idx, closest_dist

def find_closest_index_heuristic(x_waypoints, y_waypoints, vehicle_position, close_idx):
	# Find the nearest waypoint to the vehicle
    min_distance = float('inf')
    nearest_waypoint_index = close_idx
    length = len(x_waypoints)

    for i in range(len(x_waypoints)):
        j = (i + close_idx) % length
        distance = math.sqrt((vehicle_position[0] - x_waypoints[j]) ** 2 + (vehicle_position[1] - y_waypoints[j]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_waypoint_index = j
        else:
            break

    return nearest_waypoint_index

def get_waypoints_ahead(x_waypoints, y_waypoints, vehicle_position, L):
    # Find the nearest waypoint to the vehicle
    min_distance = math.inf
    nearest_waypoint_index = 0

    for i in range(len(x_waypoints)):
        distance = math.sqrt((vehicle_position[0] - x_waypoints[i]) ** 2 + (vehicle_position[1] - y_waypoints[i]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_waypoint_index = i

    # Iterate through waypoints from the nearest waypoint and accumulate distances
    accumulated_distance = 0.0
    waypoints_ahead_x = []
    waypoints_ahead_y = []

    for i in range(nearest_waypoint_index, len(x_waypoints)):
        distance = math.sqrt((vehicle_position[0] - x_waypoints[i]) ** 2 + (vehicle_position[1] - y_waypoints[i]) ** 2)
        accumulated_distance += distance
        waypoints_ahead_x.append(x_waypoints[i])
        waypoints_ahead_y.append(y_waypoints[i])

        # Break the loop if accumulated distance exceeds the lookahead distance
        if accumulated_distance >= L:
            break

    return waypoints_ahead_x, waypoints_ahead_y


def get_waypoints_ahead_looped(x_waypoints, y_waypoints, vehicle_position, L):
    # Find the nearest waypoint to the vehicle
    min_distance = math.inf
    nearest_waypoint_index = 0

    for i in range(len(x_waypoints)):
        distance = math.sqrt((vehicle_position[0] - x_waypoints[i]) ** 2 + (vehicle_position[1] - y_waypoints[i]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_waypoint_index = i

    # Iterate through waypoints from the nearest waypoint and accumulate distances
    accumulated_distance = 0.0
    waypoints_ahead_x = []
    waypoints_ahead_y = []

    for i in range(nearest_waypoint_index, nearest_waypoint_index + len(x_waypoints)):
        index = i % len(x_waypoints)  # Wrap around the waypoints list

        distance = math.sqrt((vehicle_position[0] - x_waypoints[index]) ** 2 + (vehicle_position[1] - y_waypoints[index]) ** 2)
        accumulated_distance += distance
        waypoints_ahead_x.append(x_waypoints[index])
        waypoints_ahead_y.append(y_waypoints[index])

        # Break the loop if accumulated distance exceeds the lookahead distance
        if accumulated_distance >= L:
            break

    return waypoints_ahead_x, waypoints_ahead_y, accumulated_distance

def get_waypoints_heuristic(x_waypoints, y_waypoints, vehicle_position, L, prev_idx):
    # Find the nearest waypoint to the vehicle
    min_distance = float('inf')
    nearest_waypoint_index = 0
    length = len(x_waypoints)

    for i in range(len(x_waypoints)):
        j = (i + prev_idx) % length
        distance = math.sqrt((vehicle_position[0] - x_waypoints[j]) ** 2 + (vehicle_position[1] - y_waypoints[j]) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_waypoint_index = j
        else:
            break

    # Iterate through waypoints from the nearest waypoint and accumulate distances
    accumulated_distance = 0.0
    waypoints_ahead_x = []
    waypoints_ahead_y = []

    for i in range(nearest_waypoint_index, nearest_waypoint_index + len(x_waypoints)):
        index = i % len(x_waypoints)  # Wrap around the waypoints list

        if i == nearest_waypoint_index:
            distance = math.sqrt((vehicle_position[0] - x_waypoints[index]) ** 2 + (vehicle_position[1] - y_waypoints[index]) ** 2)
        else:
            distance = math.sqrt((x_waypoints[index-1] - x_waypoints[index]) ** 2 + (y_waypoints[index-1] - y_waypoints[index]) ** 2)
        accumulated_distance += distance
        waypoints_ahead_x.append(x_waypoints[index])
        waypoints_ahead_y.append(y_waypoints[index])

        # Break the loop if accumulated distance exceeds the lookahead distance
        if accumulated_distance >= L:
            break

    return waypoints_ahead_x, waypoints_ahead_y, accumulated_distance, nearest_waypoint_index

def get_ego_trajectory(x_waypoints, y_waypoints, vehicle_position, prev_idx, velocity, time_horizon):
    cur_index = find_closest_index_heuristic(x_waypoints, y_waypoints, vehicle_position, prev_idx)
    length = len(x_waypoints)
    ego_trajectory_x = []
    ego_trajectory_y = []

    timesteps = np.arange(0.1, time_horizon + 0.1, 0.1)
    for timestep in timesteps:
        distance = velocity * timestep
        index_shift = ((int(distance/0.01)) + prev_idx) % length
        ego_trajectory_x.append(x_waypoints[index_shift])
        ego_trajectory_y.append(y_waypoints[index_shift])

    return ego_trajectory_x, ego_trajectory_y, cur_index

def get_arc_radius(wp, dist, rob_pos, rob_head):
    p = transform_point_to_local_frame(wp, rob_pos, rob_head)
    r = dist ** 2 / (2 * p[1])
    return r


def get_arc_curvature(wp, dist, rob_pos, rob_head):
    return 1 / get_arc_radius(wp, dist, rob_pos, rob_head)


def load_tack_txt(file):
    with open(file) as f:
        for i, line in enumerate(f.readlines()):
            if i == 1:
                x_str = line
                x_list = x_str.split(',')
                x = [float(xi) for xi in x_list]
            elif i == 2:
                y_str = line
                y_list = y_str.split(',')
                y = [float(yi) for yi in y_list]
    return x, y


def plot_sim(x, y, x_track, y_track):
    plt.subplot(3, 1, 1)
    plt.plot(x, y, 'k')
    plt.plot(x_track, y_track, 'r')
    plt.axis('equal')
    plt.title('trajectory')
    plt.subplot(3, 1, 2)
    plt.plot(range(len(x_track)), x_track)
    plt.title('x')
    plt.subplot(3, 1, 3)
    plt.plot(range(len(y_track)), y_track)
    plt.title('y')
    plt.show()

    plt.plot(x, y, 'k')
    plt.plot(x_track, y_track, 'r')
    plt.axis('equal')
    plt.title('trajectory')
    plt.show()

def main():
    file = os.path.join('tracks', 'oval_track_single_centerline5.txt')
    x, y = load_tack_txt(file)
    x0, y0 = x[0], y[0]
    x1, y1 = x[1], y[1]
    dyn = 'kinematicBicycle'  # 'teleport', 'singleIntegrator', 'unicycle', 'singleTrack'

    v = 1
    kp = 1
    theta0 = np.arctan2(y1 - y0, x1 - x0)
    lookahead = 0.5
    dt = 0.05
    steer_min, steer_max = -np.pi / 5, np.pi / 5
    sigma_delta, sigma_a = 0.1, 0.5

    # create robot object
    if dyn == 'teleport':
        robot = Teleport(x0, y0)
    elif dyn == 'singleIntegrator':
        robot = SingleIntegrator(x0, y0, dt)
    elif dyn == 'unicycle':
        robot = Unicycle(x0, y0, theta0, v, dt)
    elif dyn == 'singleTrack':
        robot = SingleTrack(x=x0, y=y0, heading=theta0, v=v, lr=0.1, lf=0.1, dt=dt)
    elif dyn == 'kinematicBicycle':
        robot = KinematicBicycle(x=x0, y=y0, heading=theta0, v=v, lr=0.3, lf=0.1, max_steer=0.6283, dt=dt)
    else:
        raise NotImplementedError('Undefined dynamics type')

    print(robot)

    x_trajs, y_trajs = [], []

    plt.plot(x, y, 'k')
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    l_c = 6

    for sim in range(11):

        robot = KinematicBicycle(x=x0, y=y0, heading=theta0, v=v, lr=0.3, lf=0.1, max_steer=0.6283, dt=dt)

        x_track, y_track = [x0], [y0]

        # simulate the model for 10 time steps
        for i in range(250):
            w1 = 0 # (np.random.rand() - 0.5) / 10
            w2 = 0 # (np.random.rand() - 0.5) / 10
            robot_pos = np.array([robot.x + w1, robot.y + w2])
            waypoints_ahead_x, waypoints_ahead_y, dist = get_waypoints_ahead_looped(x, y, robot_pos, lookahead)
            wp = np.array([waypoints_ahead_x[-1], waypoints_ahead_y[-1]])

            if dyn == 'teleport':
                robot.update(wp[0], wp[1])
            elif dyn == 'singleIntegrator':
                u = (wp - robot_pos) / dt
                robot.update(u)
            elif dyn == 'unicycle':
                w = kp * get_arc_curvature(wp, dist, robot_pos, robot.theta) / dt
                robot.update(v, w)
            elif dyn == 'singleTrack':
                accel = 0
                steer_angle = kp * get_arc_curvature(wp, dist, robot_pos, robot.heading)
                steer_angle = np.clip(steer_angle, steer_min, steer_max)
                u = [accel, steer_angle]
                robot.update(u)
            elif dyn == 'kinematicBicycle':
                accel = 0
                steer_angle = kp * get_arc_curvature(wp, dist, robot_pos, robot.heading)
                steer_angle = np.clip(steer_angle, steer_min, steer_max)
                u = [accel + np.random.normal(0, sigma_a), steer_angle + np.random.normal(0, sigma_delta)]
                if sim == 10:
                    u = [accel, steer_angle]
                robot.update(u)
            else:
                raise NotImplementedError('Undefined dynamics type')

            x_track.append(robot.x)
            y_track.append(robot.y)
            # print(robot)

        x_trajs.append(x_track)
        y_trajs.append(y_track)

        if sim != 10:
            plt.plot(x_track, y_track, color=colors[sim % l_c], alpha=0.7)

        # if sim == 1 or sim == 9:
        #     plot_sim(x, y, x_track, y_track)

    # print(x_trajs)
    # print(y_trajs)
    print(x)
    print(y)
    plt.axis('equal')
    plt.title('trajectory')
    plt.show()

    plt.plot(x, y, 'k')
    plt.plot(x_trajs[-1], y_trajs[-1], color='r', alpha=0.7)
    plt.axis('equal')
    plt.title('trajectory')
    plt.show()


if __name__ == "__main__":
    main()
