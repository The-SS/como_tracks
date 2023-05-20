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
    x_right_inner_centerline = L / 2.0 + R * np.cos(theta_right_inner_centerline)
    y_right_inner_centerline = R * np.sin(theta_right_inner_centerline)
    x_left_inner_centerline = -L / 2.0 - R * np.cos(theta_left_inner_centerline)
    y_left_inner_centerline = R * np.sin(theta_left_inner_centerline)

    # Define x and y values for straight edges
    x_top_inner_centerline = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    y_top_inner_centerline = np.ones(len(x_top_inner_centerline)) * R
    x_bot_inner_centerline = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    y_bot_inner_centerline = np.ones(len(x_top_inner_centerline)) * - R

    # Combine x and y values
    x_inner_centerline = np.concatenate((x_top_inner_centerline, x_left_inner_centerline, x_bot_inner_centerline, x_right_inner_centerline))
    y_inner_centerline = np.concatenate((y_top_inner_centerline, y_left_inner_centerline, y_bot_inner_centerline, y_right_inner_centerline))
    traj_inner = np.vstack([x_inner_centerline, y_inner_centerline])

    """ 
    OUTER CENTERLINE
    """
    R_outer_centerline = R+1.4

    # Define x-y values for circular caps
    theta_right_outer_centerline = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R_outer_centerline / ds)))
    theta_left_outer_centerline = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R_outer_centerline / ds)))
    x_right_outer_centerline = L / 2.0 + R_outer_centerline * np.cos(theta_right_outer_centerline)
    y_right_outer_centerline = R_outer_centerline * np.sin(theta_right_outer_centerline)
    x_left_outer_centerline = -L / 2.0 - R_outer_centerline * np.cos(theta_left_outer_centerline)
    y_left_outer_centerline = R_outer_centerline * np.sin(theta_left_outer_centerline)

    # Define x and y values for straight edges
    x_top_outer_centerline = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    y_top_outer_centerline = np.ones(len(x_top_outer_centerline)) * R_outer_centerline
    x_bot_outer_centerline = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    y_bot_outer_centerline = np.ones(len(x_top_outer_centerline)) * - R_outer_centerline

    # Combine x and y values
    x_outer_centerline = np.concatenate((x_top_outer_centerline, x_left_outer_centerline, x_bot_outer_centerline, x_right_outer_centerline))
    y_outer_centerline = np.concatenate((y_top_outer_centerline, y_left_outer_centerline, y_bot_outer_centerline, y_right_outer_centerline))
    traj_outer = np.vstack([x_outer_centerline, y_outer_centerline])

    """ 
    INNER BORDER
    """
    R_inner_border = R-0.7
    # Define x-y values for INNER circular caps
    theta_right_inner = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R_inner_border / ds)))
    theta_left_inner = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R_inner_border / ds)))
    x_right_inner = L / 2.0 + R_inner_border * np.cos(theta_right_inner)
    y_right_inner = R_inner_border * np.sin(theta_right_inner)
    x_left_inner = -L / 2.0 - R_inner_border * np.cos(theta_left_inner)
    y_left_inner = R_inner_border * np.sin(theta_left_inner)

    # Define x and y values for INNER straight edges
    x_top_inner = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    y_top_inner = np.ones(len(x_top_inner)) * R_inner_border
    x_bot_inner = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    y_bot_inner = np.ones(len(x_top_inner)) * - R_inner_border

    # Combine INNER x and y values
    x_inner_border = np.concatenate((x_top_inner, x_left_inner, x_bot_inner, x_right_inner))
    y_inner_border = np.concatenate((y_top_inner, y_left_inner, y_bot_inner, y_right_inner))

    """ 
    OUTER BORDER
    """
    R_outer_border = R+2.1

    # Define x and y values for OUTER straight edges
    x_top_outer = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    y_top_outer = np.ones(len(x_top_outer)) * R_outer_border
    x_bot_outer = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    y_bot_outer = np.ones(len(x_top_outer)) * - R_outer_border

    # Define x-y values for OUTER circular caps
    theta_right_outer = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R_outer_border / ds)))
    theta_left_outer = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R_outer_border / ds)))
    x_right_outer = L / 2.0 + R_outer_border * np.cos(theta_right_outer)
    y_right_outer = R_outer_border * np.sin(theta_right_outer)
    x_left_outer = -L / 2.0 - R_outer_border * np.cos(theta_left_outer)
    y_left_outer = R_outer_border * np.sin(theta_left_outer)

    # Combine OUTER x and y values
    x_outer_border = np.concatenate((x_top_outer, x_left_outer, x_bot_outer, x_right_outer))
    y_outer_border = np.concatenate((y_top_outer, y_left_outer, y_bot_outer, y_right_outer))

    """ 
    LANE BORDER
    """
    R_lane = R+0.7

    # Define x and y values for LANE straight edges
    x_top_lane = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    y_top_lane = np.ones(len(x_top_lane)) * R_lane
    x_bot_lane = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    y_bot_lane = np.ones(len(x_top_lane)) * - R_lane

    # Define x-y values for LANE circular caps
    theta_right_lane = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R_lane / ds)))
    theta_left_lane = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R_lane / ds)))
    x_right_lane = L / 2.0 + R_lane * np.cos(theta_right_lane)
    y_right_lane = R_lane * np.sin(theta_right_lane)
    x_left_lane = -L / 2.0 - R_lane * np.cos(theta_left_lane)
    y_left_lane = R_lane * np.sin(theta_left_lane)

    # Combine LANE x and y values
    x_lane = np.concatenate((x_top_lane, x_left_lane, x_bot_lane, x_right_lane))
    y_lane = np.concatenate((y_top_lane, y_left_lane, y_bot_lane, y_right_lane))



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

        print('\nouter traj dims: ', traj_outer.shape)
        min_dist = compute_max_distance(x_outer_centerline, y_outer_centerline)
        print("Minimum distance between successive points:", min_dist)
        min_dist = compute_max_distance(x_top_outer_centerline, y_top_outer_centerline)
        print("Minimum distance in top edge:", min_dist)
        min_dist = compute_max_distance(x_bot_outer_centerline, y_bot_outer_centerline)
        print("Minimum distance in bottom edge:", min_dist)
        min_dist = compute_max_distance(x_right_outer_centerline, y_right_outer_centerline)
        print("Minimum distance in right curve:", min_dist)
        min_dist = compute_max_distance(x_left_outer_centerline, y_left_outer_centerline)
        print("Minimum distance in left curve:", min_dist)
        print('track dims: (w, h):', max_track_pts(x_outer_centerline, y_outer_centerline))

    if show:
        # Plot oval track
        plt.scatter(x_inner_border, y_inner_border, color='black')
        plt.scatter(x_outer_border, y_outer_border, color='black')
        plt.scatter(x_lane,y_lane, linestyle='dashed', color='black')
        plt.scatter(x_inner_centerline, y_inner_centerline, color='green')
        plt.scatter(x_outer_centerline, y_outer_centerline, color='blue')
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
        plt.legend(['top', 'bot', 'right', 'left', 'top start', 'bot start', 'right start', 'left start'])
        plt.axis('equal')
        plt.title('Inner Centerline plots')
        plt.show()

        plt.subplot(2, 1, 1)
        plt.plot(x_outer_centerline, y_outer_centerline, 'k')
        plt.subplot(2, 1, 2)
        plt.plot(x_top_outer_centerline, y_top_outer_centerline)
        plt.plot(x_bot_outer_centerline, y_bot_outer_centerline)
        plt.plot(x_right_outer_centerline, y_right_outer_centerline)
        plt.plot(x_left_outer_centerline, y_left_outer_centerline)
        plt.scatter(x_top_outer_centerline[0], y_top_outer_centerline[0])
        plt.scatter(x_bot_outer_centerline[0], y_bot_outer_centerline[0])
        plt.scatter(x_right_outer_centerline[0], y_right_outer_centerline[0])
        plt.scatter(x_left_outer_centerline[0], y_left_outer_centerline[0])
        plt.legend(['top', 'bot', 'right', 'left', 'top start', 'bot start', 'right start', 'left start'])
        plt.axis('equal')
        plt.title('Outer Centerline plots')
        plt.show()

    if save:
        if save_path is not None:
            path_exists = os.path.exists(save_path)
            if not path_exists:
                os.makedirs(save_path)
            file = os.path.join(save_path, filename)
        else:
            file = filename

        #total_distance = 2*L+(2*np.pi*R_lane)
        #print(total_distance)

        coord_rows = [x_inner_border, y_inner_border, x_inner_centerline, y_inner_centerline, x_lane, y_lane, x_outer_centerline, y_outer_centerline,
                  x_outer_border, y_outer_border, x_lane, y_lane]

        with open(file, 'w', newline='') as f:
            coord_writer = csv.writer(f)
            #coord_writer.writerow('total distance is ')
            coord_writer.writerow(['edge1_x', 'edge2_y', 'centerline1_x', 'centerline1_y', 'edge2_x', 'edge2_y', 'centerline2_x', 'centerline2_y', 
                          'edge3_x', 'edge3_y', 'road_centerline_x', 'road_centerline_y'])
            for row in zip_longest(*coord_rows):
                coord_writer.writerow(row)
            
            


if __name__ == "__main__":
    # Define the oval track
    oval_track(L=5., R=2., ds=0.01, verbose=True, show=True,
               save=True, save_path='tracks', filename='oval_track_two_centerline.csv')

