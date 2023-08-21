"""
Author: Hussein Jabak 
Email:  hjabak99@gmail.com
Date:   08/06/2023
"""

import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
from scipy.interpolate import interp1d

def generate_infinity_symbol_coordinates(a, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_infinity = a * np.cos(theta) / (1 + np.sin(theta)**2)
    y_infinity = a * np.cos(theta) * np.sin(theta) / (1 + np.sin(theta)**2)
    return x_infinity, y_infinity

def calculate_normals(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    magnitudes = np.sqrt(dx_dt**2 + dy_dt**2)

    # Calculate the normals
    nx = dy_dt / magnitudes
    ny = -dx_dt / magnitudes

    return nx, ny

def calculate_normals_flipped(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    magnitudes = np.sqrt(dx_dt**2 + dy_dt**2)

    # Calculate the normals
    nx = -dy_dt / magnitudes
    ny = dx_dt / magnitudes

    return nx, ny

def calculate_cumulative_distance(x_coords, y_coords):
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    distances = np.sqrt(dx ** 2 + dy ** 2)
    cumulative_distances = np.cumsum(distances)
    return cumulative_distances

def apply_discrete_step(x_coords, y_coords, discretestep):
    """
    Parameters:
    x_coords: List or numpy array of x coordinates
    y_coords: List or numpy array of y coordinates
    discretestep: Desired distance between any two consecutive points

    Returns:
    x_discrete: List of x coordinates after spacing
    y_discrete: List of y coordinates after spacing
    """
    num_points = len(x_coords)
    
    # Calculate cumulative distances
    cumulative_distances = np.zeros(num_points)
    for i in range(1, num_points):
        dx = x_coords[i] - x_coords[i - 1]
        dy = y_coords[i] - y_coords[i - 1]
        cumulative_distances[i] = cumulative_distances[i - 1] + np.sqrt(dx**2 + dy**2)
    
    # Create interpolation functions
    f_x = interp1d(cumulative_distances, x_coords, kind='linear')
    f_y = interp1d(cumulative_distances, y_coords, kind='linear')
    
    # Generate new points
    num_steps = int(cumulative_distances[-1] / discretestep)
    new_cumulative_distances = np.linspace(0, cumulative_distances[-1], num_steps)
    x_discrete = f_x(new_cumulative_distances)
    y_discrete = f_y(new_cumulative_distances)
    
    return x_discrete, y_discrete

def find_intersection_or_similar_points(line1_x, line1_y, line2_x, line2_y):
    """
    Find the intersection between the stopping border(line1), and the centerline(line2).

    Parameters:
    line1_x, line1_y: Lists of x and y coordinates for line 1
    line2_x, line2_y: Lists of x and y coordinates for line 2
    """
    def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
        # Check for vertical lines
        if x1 == x2 and x3 == x4:
            if x1 == x3 and min(y1, y2) <= y3 <= max(y1, y2):
                return [x1], [y3]
            else:
                return [], []
        elif x1 == x2:
            x = x1
            slope2 = (y4 - y3) / (x4 - x3)
            y = slope2 * (x - x3) + y3
            if min(y1, y2) <= y <= max(y1, y2):
                return [x], [y]
            else:
                return [], []
        elif x3 == x4:
            x = x3
            slope1 = (y2 - y1) / (x2 - x1)
            y = slope1 * (x - x1) + y1
            if min(y3, y4) <= y <= max(y3, y4):
                return [x], [y]
            else:
                return [], []
        else:
            # Calculate slopes of the lines
            slope1 = (y2 - y1) / (x2 - x1)
            slope2 = (y4 - y3) / (x4 - x3)

            # Check if the lines are parallel or coincident
            if slope1 == slope2:
                return [], []

            # Calculate intersection point using the slopes and point-slope form of a line
            x = (slope1 * x1 - slope2 * x3 + y3 - y1) / (slope1 - slope2)
            y = slope1 * (x - x1) + y1

            # Check if the intersection point lies on both line segments
            if (min(x1, x2) <= x <= max(x1, x2)) and (min(x3, x4) <= x <= max(x3, x4)):
                return [x], [y]
            else:
                return [], []

    def find_similar_point(x1, y1, x2, y2, x3, y3, x4, y4):
        # Check if the lines have a similar point
        if x1 == x2 and y1 == y2:
            return [x1], [y1]
        else:
            return [], []

    def find_closest_points(x_points, y_points, x_centerline, y_centerline):
        closest_x = []
        closest_y = []

        for x, y in zip(x_points, y_points):
            min_distance = float('inf')
            closest_x_point, closest_y_point = None, None

            for cx, cy in zip(x_centerline, y_centerline):
                distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_x_point = cx
                    closest_y_point = cy

            closest_x.append(closest_x_point)
            closest_y.append(closest_y_point)

        return closest_x, closest_y

    x_points = []
    y_points = []

    # Iterate through each segment of line 1
    for i in range(len(line1_x) - 1):
        x1, y1 = line1_x[i], line1_y[i]
        x2, y2 = line1_x[i + 1], line1_y[i + 1]

        # Iterate through each segment of line 2
        for j in range(len(line2_x) - 1):
            x3, y3 = line2_x[j], line2_y[j]
            x4, y4 = line2_x[j + 1], line2_y[j + 1]

            # Check if the segments intersect and find the intersection point
            intersections_x, intersections_y = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            x_points.extend(intersections_x)
            y_points.extend(intersections_y)

            # Check if the segments share a similar point
            similar_x, similar_y = find_similar_point(x1, y1, x2, y2, x3, y3, x4, y4)
            x_points.extend(similar_x)
            y_points.extend(similar_y)

    # Find the closest points in the centerline
    closest_x, closest_y = find_closest_points(x_points, y_points, line2_x, line2_y)

    return closest_x, closest_y

def find_index_on_centerline(x, y, centerline_x, centerline_y):
    min_distance = float('inf')
    closest_index = None

    for i in range(len(centerline_x)):
        distance = (x - centerline_x[i])**2 + (y - centerline_y[i])**2
        if distance < min_distance:
            min_distance = distance
            closest_index = i

    return closest_index

def figure8_track(I: float, B: float, ds: float, num_points= 1000, show=True, save=True, save_path=None, filename='track.csv'):
    """
    Defines a figure eight track with two lanes defined as the centerlines.
    :param I: Size of Infinity symbol (meters)
    :parap B: Distance between centerline and border on either side (meters)
    :param ds: discrete step size (meters) (change ds to change track density)
    :param num_points: density for inner track creation
    :param show: True --> plots the track
    :param save: True --> saves the track in a text file
    :return:
    """

    """
    GENERATE LEFT SIDE OF FIGURE EIGHT TRACK
    """
    # Generate coordinates for the infinity symbol
    x_border_1, y_border_1 = generate_infinity_symbol_coordinates(I, num_points)

    # Filter points with x > 0
    mask = x_border_1 < 0
    x_border_1 = x_border_1[mask]
    y_border_1 = y_border_1[mask]

    # Calculate the normal vectors to the infinity symbol
    nx, ny = calculate_normals(x_border_1, y_border_1)
    x_centerline1 = x_border_1 - B * nx
    y_centerline1 = y_border_1 - B * ny

    nx, ny = calculate_normals(x_centerline1, y_centerline1)
    x_lane_border = x_centerline1 - B * nx
    y_lane_border = y_centerline1 - B * ny

    nx, ny = calculate_normals(x_lane_border, y_lane_border)
    x_centerline2 = x_lane_border - B * nx
    y_centerline2 = y_lane_border - B * ny

    nx, ny = calculate_normals(x_centerline2, y_centerline2)
    x_border_2 = x_centerline2 - B * nx
    y_border_2 = y_centerline2 - B * ny

    """
    GENERATE RIGHT SIDE OF FIGURE EIGHT TRACK
    """
    # Generate flipped coordinates for the infinity symbol
    x_border_1_flip = -x_border_1
    y_border_1_flip = y_border_1

    # Calculate the normal vectors to the flipped infinity symbol
    nx, ny = calculate_normals_flipped(x_border_1_flip, y_border_1_flip)
    x_centerline1_flip = x_border_1_flip - B * nx
    y_centerline1_flip = y_border_1_flip - B * ny

    nx, ny = calculate_normals_flipped(x_centerline1_flip, y_centerline1_flip)
    x_lane_border_flip = x_centerline1_flip - B * nx
    y_lane_border_flip = y_centerline1_flip - B * ny

    nx, ny = calculate_normals_flipped(x_lane_border_flip, y_lane_border_flip)
    x_centerline2_flip = x_lane_border_flip - B * nx
    y_centerline2_flip = y_lane_border_flip - B * ny

    nx, ny = calculate_normals_flipped(x_centerline2_flip, y_centerline2_flip)
    x_border_2_flip = x_centerline2_flip - B * nx
    y_border_2_flip = y_centerline2_flip - B * ny

    """
    SHIFT TRACKS ACCORDINGLY
    """
    # Get the shift value for the original lists
    x_shift = x_border_2[0]

    # Move the original points by the shift value
    x_border_1_shifted = x_border_1 - x_shift
    x_centerline1_shifted = x_centerline1 - x_shift
    x_lane_border_shifted = x_lane_border - x_shift
    x_centerline2_shifted = x_centerline2 - x_shift
    x_border_2_shifted = x_border_2 - x_shift

    # Get the shift value for the flipped lists
    x_shift_flip = x_border_2_flip[0]

    # Move the flipped points by the shift value
    x_border_1_flip_shifted = x_border_1_flip - x_shift_flip
    x_centerline1_flip_shifted = x_centerline1_flip - x_shift_flip
    x_lane_border_flip_shifted = x_lane_border_flip - x_shift_flip
    x_centerline2_flip_shifted = x_centerline2_flip - x_shift_flip
    x_border_2_flip_shifted = x_border_2_flip - x_shift_flip
    
    """
    CONNECT BOTH CURVES WITH STRAIGHT LINES TO FORM FIGURE EIGHT
    """
    # Create straight lines with the same point density as the existing lists
    num_points_straight_lines = len(x_centerline1)

    # Generate the straight lines for centerline 1
    x_straight_line1 = np.linspace(x_centerline1_shifted[-1], x_centerline2_flip_shifted[0], num_points_straight_lines)
    y_straight_line1 = np.linspace(y_centerline1[-1], y_centerline2_flip[0], num_points_straight_lines)

    x_straight_line2 = np.linspace(x_centerline2_flip_shifted[-1], x_centerline1_shifted[0], num_points_straight_lines)
    y_straight_line2 = np.linspace(y_centerline2_flip[-1], y_centerline1[0], num_points_straight_lines)

    # Concatenate the straight lines
    x_concatenated_centerline1 = np.concatenate([x_centerline1_shifted, x_straight_line1, x_centerline2_flip_shifted, x_straight_line2])
    y_concatenated_centerline1 = np.concatenate([y_centerline1, y_straight_line1, y_centerline2_flip, y_straight_line2])

    # Generate the straight lines for centerline 2
    x_straight_line1 = np.linspace(x_centerline2_shifted[-1], x_centerline1_flip_shifted[0], num_points_straight_lines)
    y_straight_line1 = np.linspace(y_centerline2[-1], y_centerline1_flip[0], num_points_straight_lines)

    x_straight_line2 = np.linspace(x_centerline1_flip_shifted[-1], x_centerline2_shifted[0], num_points_straight_lines)
    y_straight_line2 = np.linspace(y_centerline1_flip[-1], y_centerline2[0], num_points_straight_lines)

    # Concatenate the straight lines
    x_concatenated_centerline2 = np.concatenate([x_centerline2_shifted, x_straight_line1, x_centerline1_flip_shifted, x_straight_line2])
    y_concatenated_centerline2 = np.concatenate([y_centerline2, y_straight_line1, y_centerline1_flip, y_straight_line2])


    # Generate the straight lines for lane border
    x_straight_line1 = np.linspace(x_lane_border_shifted[-1], x_lane_border_flip_shifted[0], num_points_straight_lines)
    y_straight_line1 = np.linspace(y_lane_border[-1], y_lane_border_flip[0], num_points_straight_lines)

    x_straight_line2 = np.linspace(x_lane_border_flip_shifted[-1], x_lane_border_shifted[0], num_points_straight_lines)
    y_straight_line2 = np.linspace(y_lane_border_flip[-1], y_lane_border[0], num_points_straight_lines)

    # Concatenate the straight lines
    x_concatenated_lane_border = np.concatenate([x_lane_border_shifted, x_straight_line1, x_lane_border_flip_shifted, x_straight_line2])
    y_concatenated_lane_border = np.concatenate([y_lane_border, y_straight_line1, y_lane_border_flip, y_straight_line2])


    # Generate the straight lines for border 1
    x_straight_line1 = np.linspace(x_border_1_shifted[-1], x_border_2_flip_shifted[0], num_points_straight_lines)
    y_straight_line1 = np.linspace(y_border_1[-1], y_border_2_flip[0], num_points_straight_lines)

    x_border_1_intersection = np.linspace(x_border_2_flip_shifted[-1], x_border_1_shifted[0], num_points_straight_lines)
    y_border_1_intersection = np.linspace(y_border_2_flip[-1], y_border_1[0], num_points_straight_lines)

    # Concatenate the straight lines
    x_concatenated_border_1 = np.concatenate([x_border_1_shifted, x_straight_line1, x_border_2_flip_shifted, x_border_1_intersection])
    y_concatenated_border_1 = np.concatenate([y_border_1, y_straight_line1, y_border_2_flip, y_border_1_intersection])


    # Generate the straight lines for border 2
    x_border_2_intersection = np.linspace(x_border_2_shifted[-1], x_border_1_flip_shifted[0], num_points_straight_lines)
    y_border_2_intersection = np.linspace(y_border_2[-1], y_border_1_flip[0], num_points_straight_lines)

    x_straight_line2 = np.linspace(x_border_1_flip_shifted[-1], x_border_2_shifted[0], num_points_straight_lines)
    y_straight_line2 = np.linspace(y_border_1_flip[-1], y_border_2[0], num_points_straight_lines)

    # Concatenate the straight lines
    x_concatenated_border_2 = np.concatenate([x_border_2_shifted, x_border_2_intersection, x_border_1_flip_shifted, x_straight_line2])
    y_concatenated_border_2 = np.concatenate([y_border_2, y_border_2_intersection, y_border_1_flip, y_straight_line2])

    """
    FIND STOPPING POINTS FOR CENTERLINES
    """
    x1_stopping, y1_stopping = find_intersection_or_similar_points(x_border_1_intersection, y_border_1_intersection, x_concatenated_centerline1, y_concatenated_centerline1)
    x1_stopping2, y1_stopping2 = find_intersection_or_similar_points(x_border_2_intersection, y_border_2_intersection, x_concatenated_centerline1, y_concatenated_centerline1)
    x2_stopping, y2_stopping = find_intersection_or_similar_points(x_border_1_intersection, y_border_1_intersection, x_concatenated_centerline2, y_concatenated_centerline2)
    x2_stopping2, y2_stopping2 = find_intersection_or_similar_points(x_border_2_intersection, y_border_2_intersection, x_concatenated_centerline2, y_concatenated_centerline2)

    x1_stopping += x1_stopping2
    y1_stopping += y1_stopping2
    x2_stopping += x2_stopping2
    y2_stopping += y2_stopping2

    """
    CHECK THROUGH LISTS AND ADJUST SPACE BETWEEN POINTS
    """
    x_concatenated_centerline1, y_concatenated_centerline1 = apply_discrete_step(x_concatenated_centerline1, y_concatenated_centerline1, ds)
    x_concatenated_centerline2, y_concatenated_centerline2 = apply_discrete_step(x_concatenated_centerline2, y_concatenated_centerline2, ds)
    x_concatenated_lane_border, y_concatenated_lane_border = apply_discrete_step(x_concatenated_lane_border, y_concatenated_lane_border, ds)
    x_concatenated_border_1,    y_concatenated_border_1    = apply_discrete_step(x_concatenated_border_1,    y_concatenated_border_1, ds)
    x_concatenated_border_2,    y_concatenated_border_2    = apply_discrete_step(x_concatenated_border_2,    y_concatenated_border_2, ds)
    x_border_1_intersection,    y_border_1_intersection    = apply_discrete_step(x_border_1_intersection,    y_border_1_intersection, ds)
    x_border_2_intersection,    y_border_2_intersection    = apply_discrete_step(x_border_2_intersection,    y_border_2_intersection, ds)

    """
    MOVE STOPPING POINT AND STOPPING BORDER AWAY FROM INTERSECTION
    """
    move_by = 25

    index1 = find_index_on_centerline(x1_stopping[0], y1_stopping[0], x_concatenated_centerline1, y_concatenated_centerline1)
    index1 -=move_by
    x1_stopping[0] = x_concatenated_centerline1[index1]
    y1_stopping[0] = y_concatenated_centerline1[index1]

    index1 = find_index_on_centerline(x1_stopping[1], y1_stopping[1], x_concatenated_centerline1, y_concatenated_centerline1)
    index1 -=move_by
    x1_stopping[1] = x_concatenated_centerline1[index1]
    y1_stopping[1] = y_concatenated_centerline1[index1]

    index1 = find_index_on_centerline(x2_stopping[0], y2_stopping[0], x_concatenated_centerline2, y_concatenated_centerline2)
    index1 -=move_by
    x2_stopping[0] = x_concatenated_centerline2[index1]
    y2_stopping[0] = y_concatenated_centerline2[index1]

    index1 = find_index_on_centerline(x2_stopping[1], y2_stopping[1], x_concatenated_centerline2, y_concatenated_centerline2)
    index1 -=move_by
    x2_stopping[1] = x_concatenated_centerline2[index1]
    y2_stopping[1] = y_concatenated_centerline2[index1]

    # Calculate the midpoint between x1_stopping[0], x2_stopping[0] and y1_stopping[0], y2_stopping[0]
    midpoint_x = (x1_stopping[0] + x2_stopping[0]) / 2
    midpoint_y = (y1_stopping[0] + y2_stopping[0]) / 2

    # Calculate the shift needed for x_border_1_intersection and y_border_1_intersection
    shift_x = midpoint_x - (x_border_1_intersection[0] + x_border_1_intersection[-1]) / 2
    shift_y = midpoint_y - (y_border_1_intersection[0] + y_border_1_intersection[-1]) / 2

    # Apply the shift to x_border_1_intersection and y_border_1_intersection
    x_border_1_intersection = [x + shift_x for x in x_border_1_intersection]
    y_border_1_intersection = [y + shift_y for y in y_border_1_intersection]
    
    # Calculate the midpoint between x1_stopping[1], x2_stopping[1] and y1_stopping[1], y2_stopping[1]
    midpoint_x = (x1_stopping[1] + x2_stopping[1]) / 2
    midpoint_y = (y1_stopping[1] + y2_stopping[1]) / 2

    # Calculate the shift needed for x_border_1_intersection and y_border_2_intersection
    shift_x = midpoint_x - (x_border_2_intersection[0] + x_border_2_intersection[-1]) / 2
    shift_y = midpoint_y - (y_border_2_intersection[0] + y_border_2_intersection[-1]) / 2

    # Apply the shift to x_border_1_intersection and y_border_2_intersection
    x_border_2_intersection = [x + shift_x for x in x_border_2_intersection]
    y_border_2_intersection = [y + shift_y for y in y_border_2_intersection]

    if show:
        # Plot both versions of the infinity symbol with border on the same graph
        plt.figure(figsize=(10, 6))

        # Original infinity symbol with border
        plt.plot(x_border_1_shifted, y_border_1, label='Border 1', color='tab:blue')
        plt.plot(x_centerline1_shifted,  y_centerline1,  label='Centerline 1', color='tab:red')
        plt.plot(x_lane_border_shifted,  y_lane_border,  label='Lane Border',  color='tab:blue')
        plt.plot(x_centerline2_shifted,  y_centerline2,  label='Centerline 2', color='tab:red')
        plt.plot(x_border_2_shifted, y_border_2, label='Border 1', color='tab:blue')

        # Plot the first point of each line with a different color and marker style
        plt.scatter(x_border_1_shifted[0], y_border_1[0], color='tab:blue', marker='o', label='Start (Border 1)')
        plt.scatter(x_centerline1_shifted[0], y_centerline1[0], color='tab:green', marker='o', label='Start (Centerline 1)')
        plt.scatter(x_lane_border_shifted[0], y_lane_border[0], color='tab:blue', marker='o', label='Start (Lane Border)')
        plt.scatter(x_centerline2_shifted[0], y_centerline2[0], color='tab:red', marker='o', label='Start (Centerline 2)')
        plt.scatter(x_border_2_shifted[0], y_border_2[0], color='tab:blue', marker='o', label='Start (Border 2)')

        # Flipped infinity symbol with border
        plt.plot(x_border_1_flip_shifted, y_border_1_flip, label='Border 1 (Flipped)', color='tab:blue', linestyle='dashed')
        plt.plot(x_centerline1_flip_shifted,  y_centerline1_flip,  label='Centerline 1 (Flipped)', color='tab:red', linestyle='dashed')
        plt.plot(x_lane_border_flip_shifted,  y_lane_border_flip,  label='Lane Border (Flipped)',  color='tab:blue', linestyle='dashed')
        plt.plot(x_centerline2_flip_shifted,  y_centerline2_flip,  label='Centerline 2 (Flipped)', color='tab:red', linestyle='dashed')
        plt.plot(x_border_2_flip_shifted, y_border_2_flip, label='Border 2 (Flipped)', color='tab:blue', linestyle='dashed')

        # Plot the first point of each line in the flipped version
        plt.scatter(x_border_1_flip_shifted[0], y_border_1_flip[0], color='tab:blue', marker='o', linestyle='dashed', label='Start (Border 1 Flipped)')
        plt.scatter(x_centerline1_flip_shifted[0], y_centerline1_flip[0], color='tab:purple', marker='o', linestyle='dashed', label='Start (Centerline 1 Flipped)')
        plt.scatter(x_lane_border_flip_shifted[0], y_lane_border_flip[0], color='tab:blue', marker='o', linestyle='dashed', label='Start (Lane Border Flipped)')
        plt.scatter(x_centerline2_flip_shifted[0], y_centerline2_flip[0], color='tab:red', marker='o', linestyle='dashed', label='Start (Centerline 2 Flipped)')
        plt.scatter(x_border_2_flip_shifted[0], y_border_2_flip[0], color='tab:blue', marker='o', linestyle='dashed', label='Start (Border 2 Flipped)')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Infinity Symbol with Border')
        plt.axis('equal')
        plt.legend()
        plt.show()



        # Plot both versions of the infinity symbol with border and the concatenated lines on the same graph
        plt.figure(figsize=(10, 6))

        # Plot the concatenated lines as straight lines
        plt.scatter(x_concatenated_centerline1, y_concatenated_centerline1, label='Concatenated Lines', color='black', linestyle='dotted')
        plt.scatter(x_concatenated_centerline2, y_concatenated_centerline2, label='Concatenated Lines', color='black', linestyle='dotted')
        plt.scatter(x_concatenated_lane_border, y_concatenated_lane_border, label='Concatenated Lines', color='black', linestyle='dotted')
        plt.scatter(x_concatenated_border_1, y_concatenated_border_1, label='Concatenated Lines', color='black', linestyle='dotted')
        plt.scatter(x_concatenated_border_2, y_concatenated_border_2, label='Concatenated Lines', color='black', linestyle='dotted')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Infinity Symbol with Border and Concatenated Lines')
        plt.axis('equal')
        plt.legend()
        plt.show()


        # Plot road graph
        plt.figure(figsize=(10, 6))

        # Plot the concatenated lines as straight lines
        plt.plot(x_concatenated_centerline1, y_concatenated_centerline1, label='Centerline 1', color='grey', linestyle='dashed')
        plt.plot(x_concatenated_centerline2, y_concatenated_centerline2, label='Centerline 2', color='grey', linestyle='dashed')
        plt.plot(x_concatenated_lane_border, y_concatenated_lane_border, label='Lane Border', color='black')
        plt.plot(x_concatenated_border_1,    y_concatenated_border_1,    label='Border 1',   color='black', linewidth = 3)
        plt.plot(x_concatenated_border_2,    y_concatenated_border_2,    label='Border 2',   color='black', linewidth = 3)
        plt.plot(x_border_1_intersection, y_border_1_intersection, label='Intersection Stop', color='red')
        plt.plot(x_border_2_intersection, y_border_2_intersection, label='Intersection Stop', color='red')
        plt.scatter(x1_stopping, y1_stopping, label='Centerline 1 Stopping Points', color='green')
        plt.scatter(x2_stopping, y2_stopping, label='Centerline 2 Stopping Points', color='blue')

        plt.title('Figure Eight Track')
        plt.axis('equal')
        plt.legend()
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

        cumulative_distance = calculate_cumulative_distance(x_concatenated_lane_border, y_concatenated_lane_border)

        coord_rows = [x_concatenated_border_1, y_concatenated_border_1, x_concatenated_centerline1, y_concatenated_centerline1, 
                      x_concatenated_lane_border, y_concatenated_lane_border, x_concatenated_centerline2, y_concatenated_centerline2,
                      x_concatenated_border_2, y_concatenated_border_2, x_concatenated_lane_border, y_concatenated_lane_border, 
                      x_border_1_intersection, y_border_1_intersection, x_border_2_intersection, y_border_2_intersection, 
                      x1_stopping, y1_stopping, x2_stopping, y2_stopping, 
                      cumulative_distance]

        with open(file, 'w', newline='') as f:
            coord_writer = csv.writer(f)
            coord_writer.writerow(['edge1_x', 'edge1_y', 'centerline1_x', 'centerline1_y', 
                                   'edge2_x', 'edge2_y', 'centerline2_x', 'centerline2_y', 
                                   'edge3_x', 'edge3_y', 'road_centerline_x', 'road_centerline_y', 
                                   'intersection_border1_x', 'intersection_border1_y','intersection_border2_x', 'intersection_border2_y', 
                                   'centerpoint1_stop_x', 'centerpoint1_stop_y', 'centerpoint2_stop_x', 'centerpoint2_stop_y',
                                   'cumulative_distance'])
            for row in zip_longest(*coord_rows):
                coord_writer.writerow(row)

if __name__ == "__main__":
    # Define the oval track
    figure8_track(I=2.0, B=0.3048, num_points=1000, ds=0.01, show=True, save=True, save_path='tracks_2', filename='figure8_two_centerline.csv')
