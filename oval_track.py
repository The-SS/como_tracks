# File creates road data for a road with a single centerline based on the radius given in the function call
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import zip_longest

def compute_max_distance(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return np.max(dist)

def max_track_pts(x, y):
    return max(x) - min(x), max(y) - min(y)

def calculate_cumulative_distance(x_coords, y_coords):
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.sqrt(dx ** 2 + dy ** 2)
    cumulative_distances = np.cumsum(distances)
    return cumulative_distances


def oval_track(L: float, R: float, ds=0.01, verbose=True, show=True, save=True, save_path=None, filename='track.csv'):
    """
    Defines an oval track with a single lane defined as the centerline.
    This consists of two straight and parallel edges connect by two half circles on the right and left.
    The total track length is L + 2R and the width is 2R.
    :param L: length of straight edges (meters)
    :param R: radius of circular sides (meters)
    :param ds: discrete step size (meters)
    :param show: True --> plots the track
    :param save: True --> saves the track in a text file
    :param verbose: True --> print some details about the track
    :return:
    """
    
    """ 
    INNER CENTERLINE
    """
    # Define x-y values for circular caps
    theta_right_inner_centerline = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R / ds)))
    theta_left_inner_centerline = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R / ds)))
    y_right_inner_centerline = L / 2.0 + R * np.cos(theta_right_inner_centerline)
    x_right_inner_centerline = R * np.sin(theta_right_inner_centerline)
    y_left_inner_centerline = -L / 2.0 - R * np.cos(theta_left_inner_centerline)
    x_left_inner_centerline = R * np.sin(theta_left_inner_centerline)

    # Define x and y values for straight edges
    y_top_inner_centerline = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    x_top_inner_centerline = np.ones(len(y_top_inner_centerline)) * R
    y_bot_inner_centerline = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    x_bot_inner_centerline = np.ones(len(y_top_inner_centerline)) * - R

    # Combine x and y values
    x_inner_centerline = np.concatenate((x_top_inner_centerline, x_left_inner_centerline, x_bot_inner_centerline, x_right_inner_centerline))
    y_inner_centerline = np.concatenate((y_top_inner_centerline, y_left_inner_centerline, y_bot_inner_centerline, y_right_inner_centerline))
    traj_inner = np.vstack([x_inner_centerline, y_inner_centerline])

    """ 
    INNER BORDER
    """
    R_inner_border = R-0.213 # meters
    # Define x-y values for INNER circular caps
    theta_right_inner = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R_inner_border / ds)))
    theta_left_inner = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R_inner_border / ds)))
    y_right_inner = L / 2.0 + R_inner_border * np.cos(theta_right_inner)
    x_right_inner = R_inner_border * np.sin(theta_right_inner)
    y_left_inner = -L / 2.0 - R_inner_border * np.cos(theta_left_inner)
    x_left_inner = R_inner_border * np.sin(theta_left_inner)

    # Define x and y values for INNER straight edges
    y_top_inner = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    x_top_inner = np.ones(len(y_top_inner)) * R_inner_border
    y_bot_inner = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    x_bot_inner = np.ones(len(y_top_inner)) * - R_inner_border

    # Combine INNER x and y values
    x_inner_border = np.concatenate((x_top_inner, x_left_inner, x_bot_inner, x_right_inner))
    y_inner_border = np.concatenate((y_top_inner, y_left_inner, y_bot_inner, y_right_inner))

    
    """ 
    OUTER BORDER
    """
    R_outer = R+0.213 # meters

    # Define x and y values for OUTER straight edges
    y_top_outer = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    x_top_outer = np.ones(len(y_top_outer)) * R_outer
    y_bot_outer = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    x_bot_outer = np.ones(len(y_top_outer)) * - R_outer

    # Define x-y values for OUTER circular caps
    theta_right_outer = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R_outer / ds)))
    theta_left_outer = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R_outer / ds)))
    y_right_outer = L / 2.0 + R_outer * np.cos(theta_right_outer)
    x_right_outer = R_outer * np.sin(theta_right_outer)
    y_left_outer = -L / 2.0 - R_outer * np.cos(theta_left_outer)
    x_left_outer = R_outer * np.sin(theta_left_outer)

    # Combine OUTER x and y values
    x_outer = np.concatenate((x_top_outer, x_left_outer, x_bot_outer, x_right_outer))
    y_outer = np.concatenate((y_top_outer, y_left_outer, y_bot_outer, y_right_outer))



    if verbose:
        print('inner traj dims: ', traj_inner.shape)
        min_dist = compute_max_distance(x_inner_centerline, y_inner_centerline)
        print("Minimum distance between successive points:", min_dist)
        min_dist = compute_max_distance(x_top_inner_centerline, y_top_inner_centerline)
        print("Minimum distance in top edge:", min_dist)
        min_dist = compute_max_distance(x_bot_inner_centerline, y_bot_inner_centerline)
        print("Minimum distance in bottom edge:", min_dist)
        min_dist = compute_max_distance(x_right_inner_centerline, y_right_inner_centerline)
        print("Minimum distance in right curve:", min_dist)
        min_dist = compute_max_distance(x_left_inner_centerline, y_left_inner_centerline)
        print("Minimum distance in left curve:", min_dist)
        print('track dims: (w, h):', max_track_pts(x_inner_centerline, y_inner_centerline))


    if show:
        # Plot oval track
        plt.scatter(x_inner_border, y_inner_border, color='black')
        plt.scatter(x_outer, y_outer, color='black')
        plt.scatter(x_inner_centerline, y_inner_centerline, color='green')
        plt.title('Scatter plot of the track points. Increase ds for a sparser track.')
        plt.show()


        plt.subplot(2, 1, 1)
        plt.plot(x_inner_centerline, y_inner_centerline, 'k')
        plt.subplot(2, 1, 2)
        plt.plot(x_top_inner_centerline, y_top_inner_centerline)
        plt.plot(x_bot_inner_centerline, y_bot_inner_centerline)
        plt.plot(x_right_inner_centerline, y_right_inner_centerline)
        plt.plot(x_left_inner_centerline, y_left_inner_centerline)
        plt.scatter(x_top_inner_centerline[0], y_top_inner_centerline[0])
        plt.scatter(x_bot_inner_centerline[0], y_bot_inner_centerline[0])
        plt.scatter(x_right_inner_centerline[0], y_right_inner_centerline[0])
        plt.scatter(x_left_inner_centerline[0], y_left_inner_centerline[0])
        plt.legend(['right', 'left', 'top', 'bot', 'right start', 'left start', 'top start', 'bot start'])
        plt.axis('equal')
        plt.title('Inner Centerline plots')
        plt.show()

        # Plot oval track that looks like a road
        plt.plot(x_inner_border, y_inner_border, color='black', linewidth = 3)
        plt.plot(x_outer, y_outer, color='black', linewidth = 3)
        plt.plot(x_inner_centerline, y_inner_centerline, color='grey', linestyle='dashed')
        plt.title('Plot of Road')
        plt.show()

    # save the coordinates for all border, centerlines, and cumulative distance to a csv 
    # file where the headers are all on one row and their respective coordinates are in the same column
    if save:
        if save_path is not None:
            path_exists = os.path.exists(save_path)
            if not path_exists:
                os.makedirs(save_path)
            file = os.path.join(save_path, filename)
        else:
            file = filename

        cumulative_distance = calculate_cumulative_distance(x_inner_centerline, y_inner_centerline)

        coord_rows = [x_inner_border, y_inner_border, x_inner_centerline, y_inner_centerline, x_outer, y_outer, cumulative_distance]

        with open(file, 'w', newline='') as f:
            coord_writer = csv.writer(f)
            coord_writer.writerow(['edge1_x', 'edge1_y', 'centerline1_x', 'centerline1_y', 'edge2_x', 'edge2_y', 'road_centerline_x', 'road_centerline_y', 'cumulative_distance'])
            for row in zip_longest(*coord_rows):
                coord_writer.writerow(row)

if __name__ == "__main__":
    # Define the oval track
    oval_track(L=5., R=2., ds=0.01, verbose=True, show=True,
               save=True, save_path='tracks', filename='oval_track.csv')