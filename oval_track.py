import os
import numpy as np
import matplotlib.pyplot as plt


def compute_max_distance(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return np.max(dist)


def max_track_pts(x, y):
    return max(x) - min(x), max(y) - min(y)


def oval_track(L: float, R: float, ds=0.01, verbose=True, show=True, save=True, save_path=None, filename='track.txt'):
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

    # Define x-y values for circular caps
    theta_right = np.linspace(-np.pi / 2, np.pi / 2, int(np.ceil(np.pi * R / ds)))
    theta_left = np.linspace(np.pi / 2, -np.pi / 2, int(np.ceil(np.pi * R / ds)))
    x_right = L / 2.0 + R * np.cos(theta_right)
    y_right = R * np.sin(theta_right)
    x_left = -L / 2.0 - R * np.cos(theta_left)
    y_left = R * np.sin(theta_left)

    # Define x and y values for straight edges
    x_top = np.linspace(L / 2, -L / 2, int(np.ceil(L / ds)))[1:-1]
    y_top = np.ones(len(x_top)) * R
    x_bot = np.linspace(-L / 2, L / 2, int(np.ceil(L / ds)))[1:-1]
    y_bot = np.ones(len(x_top)) * - R

    # Combine x and y values
    x = np.concatenate((x_top, x_left, x_bot, x_right))
    y = np.concatenate((y_top, y_left, y_bot, y_right))
    traj = np.vstack([x, y])

    if verbose:
        print('traj dims: ', traj.shape)

        min_dist = compute_max_distance(x, y)
        print("Minimum distance between successive points:", min_dist)
        min_dist = compute_max_distance(x_top, y_top)
        print("Minimum distance in top edge:", min_dist)
        min_dist = compute_max_distance(x_bot, y_bot)
        print("Minimum distance in bottom edge:", min_dist)
        min_dist = compute_max_distance(x_right, y_right)
        print("Minimum distance in right curve:", min_dist)
        min_dist = compute_max_distance(x_left, y_left)
        print("Minimum distance in left curve:", min_dist)

        print('track dims: (w, h):', max_track_pts(x, y))

    if show:
        # Plot oval track
        plt.scatter(x, y)
        plt.title('Scatter plot of the track points. Increase ds for a sparser track.')
        plt.show()
        plt.subplot(2, 1, 1)
        plt.plot(x, y, 'k')
        plt.subplot(2, 1, 2)
        plt.plot(x_top, y_top)
        plt.plot(x_bot, y_bot)
        plt.plot(x_right, y_right)
        plt.plot(x_left, y_left)
        plt.scatter(x_top[0], y_top[0])
        plt.scatter(x_bot[0], y_bot[0])
        plt.scatter(x_right[0], y_right[0])
        plt.scatter(x_left[0], y_left[0])
        plt.legend(['top', 'bot', 'right', 'left', 'top start', 'bot start', 'right start', 'left start'])
        plt.axis('equal')
        plt.show()

    if save:
        if save_path is not None:
            path_exists = os.path.exists(save_path)
            if not path_exists:
                os.makedirs(save_path)
            file = os.path.join(save_path, filename)
        else:
            file = filename
        with open(file, 'w') as f:
            f.write('Format: x, y \n')
            x_str = ','.join([str(x[i]) for i in range(len(x))])
            y_str = ','.join([str(y[i]) for i in range(len(y))])
            f.write(x_str + '\n')
            f.write(y_str)


if __name__ == "__main__":
    # Define the oval track
    oval_track(L=5., R=2., ds=0.01, verbose=True, show=False,
               save=True, save_path='tracks', filename='oval_track_single_centerline.txt')
