"""
Author: Hussein Jabak 
Email:  hjabak99@gmail.com
Date:   08/06/2023
"""

import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import csv
import math

class TrackPlotter:
    def __init__(self, lab_x, lab_y, rc_x, rc_y, csv_file):
        """
        Initializes the RoadAnimation class.

        :param lab_x: width of the lab in meters
        :param lab_y: height of the lab in meters
        :param rc_x: width of the rectangular car in meters
        :param rc_y: height of the rectangular car in meters
        :param csv_file: path to the CSV file containing the coordinates
        """
        self.lab_x = lab_x
        self.lab_y = lab_y
        self.rc_x = rc_x
        self.rc_y = rc_y
        self.coordinates = TrackUtils._extract_coordinates(csv_file)

        self.edges_x = []
        self.edges_y = []
        self.centerline_x = []
        self.centerline_y = []
        self.intersection_x = []
        self.intersection_y = []
        self.stopping_point_x = []
        self.stopping_point_y = []
        self.experiment_data_x = []
        self.experiment_data_y = []

        # Parse through coordinates dictionary and append keys to corresponding lists
        for key in self.coordinates.keys():
            if key.startswith('edge'):
                if 'x' in key:
                    self.edges_x.append(self.coordinates[key])
                if 'y' in key:
                    self.edges_y.append(self.coordinates[key])
            if key.startswith('centerline'):
                if 'x' in key:
                    self.centerline_x.append(self.coordinates[key])
                if 'y' in key:
                    self.centerline_y.append(self.coordinates[key])
            if key.startswith('intersection'):
                if 'x' in key:
                    self.intersection_x.append(self.coordinates[key])
                if 'y' in key:
                    self.intersection_y.append(self.coordinates[key])
            if key.startswith('centerpoint'):
                if 'x' in key:
                    self.stopping_point_x.append(self.coordinates[key])
                if 'y' in key:
                    self.stopping_point_y.append(self.coordinates[key])
            
    def plot_trajec(self, txt_file):
        """
        NOTE: Currently only works properly with tracks that contain one vehicle.

        Plots collected data from a text file and compares it with simulated centerline data.

        :param txt_file: path to the text file containing x, y, and heading angle data
        """

        # Plots collected data from txt file with x,y,heading_angle format, and plots data with simulated centerline data
        with open(txt_file, 'r') as file:
            lines = file.readlines()[1:]

        x_values = []
        y_values = []
        heading_angles = []

        # Write data from text file to variables
        for line in lines:
            x, y, h = line.strip().split(',')
            x_values.append(float(x))
            y_values.append(float(y))
            heading_angles.append(float(h))

        self.experiment_data_x = [x_values]
        self.experiment_data_y = [y_values]
        
        # Plot text file data
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, 'r', label='Road Edges')

        # Plot Road edges and centerlines
        for i in range(len(self.edges_x)):
            if i == 0 or i == len(self.edges_x) - 1:
                plt.plot(self.edges_x[i], self.edges_y[i], color='black', linewidth=3)
            else:
                plt.plot(self.edges_x[i], self.edges_y[i], color='black', linestyle='dashed')

        for i in range(len(self.centerline_x)):
            plt.plot(self.centerline_x[i], self.centerline_y[i], color='grey', linestyle='dashed', label='Road Centerline')

        plt.title('Plot of Road')
        plt.axis('equal')
        plt.legend()
        plt.show()

    def plot_track(self, show_plot=True):
        plt.figure(figsize=(10, 6))

        # Plot all static edges and centerlines of road
        for i in range(len(self.edges_x)):
            if i == 0 or i == len(self.edges_x) - 1:
                plt.plot(self.edges_x[i], self.edges_y[i], color='black', linewidth=3)
            else:
                plt.plot(self.edges_x[i], self.edges_y[i], color='black', linestyle='dashed')

        for i in range(len(self.centerline_x)):
            plt.plot(self.centerline_x[i], self.centerline_y[i], color='grey', linestyle='dashed')

        if self.intersection_x:
            for i in range(len(self.intersection_x)):
                plt.plot(self.intersection_x[i], self.intersection_y[i], color='red')

        if show_plot:
            plt.show()
        else:
            plt.close()

        return

    def add_translate(self, vector):
        """
        Translates the track across a vector.

        :param vector: translation vector [x, y]
        """

        # Take vector [x,y] format and move edges and centerlines accordingly
        tx, ty = vector

        # Add to x and y values of edges
        for i in range(len(self.edges_x)):
            self.edges_x[i] = [x + tx for x in self.edges_x[i]]
            self.edges_y[i] = [y + ty for y in self.edges_y[i]]

        # Add to x and y values of centerline
        for i in range(len(self.centerline_x)):
            self.centerline_x[i] = [x + tx for x in self.centerline_x[i]]
            self.centerline_y[i] = [y + ty for y in self.centerline_y[i]]

        # Add to x and y values of intersection lines
        if self.intersection_x:
            for i in range(len(self.intersection_x)):
                self.intersection_x[i] = [x + tx for x in self.intersection_x[i]]
                self.intersection_y[i] = [y + ty for y in self.intersection_y[i]]

        # Add to x and y values of stopping points
        if self.stopping_point_x:
            for i in range(len(self.stopping_point_x)):
                self.stopping_point_x[i] = [x + tx for x in self.stopping_point_x[i]]
                self.stopping_point_y[i] = [y + ty for y in self.stopping_point_y[i]]

    def rotate_track(self, angle):
        """
        Rotates the track at a given angle.

        :param angle: rotation angle in degrees
        """

        # Convert the angle from degrees to radians
        angle_rad = math.radians(angle)

        # Rotate the edges
        for i in range(len(self.edges_x)):
            self.edges_x[i], self.edges_y[i] = TrackUtils._rotate_points(self.edges_x[i], self.edges_y[i], angle_rad)

        # Rotate the centerline
        for i in range(len(self.centerline_x)):
            self.centerline_x[i], self.centerline_y[i] = TrackUtils._rotate_points(self.centerline_x[i], self.centerline_y[i], angle_rad)

        if self.intersection_x:
            for i in range(len(self.intersection_x)):
                self.intersection_x[i], self.intersection_y[i] = TrackUtils._rotate_points(self.intersection_x[i], self.intersection_y[i], angle_rad)

    def save_track(self, save_path, filename):
        """
        Saves the current track coordinates to a CSV file.

        :param save_path: path to the directory to save the file
        :param filename: name of the saved file (should end with .csv)
        """

        # Add all edge_x, edge_y, centerline_x, centerline_y numeration to header
        headers = ['edge{}_x'.format(i) for i in range(1, len(self.edges_x) + 1)]
        headers += ['edge{}_y'.format(i) for i in range(1, len(self.edges_y) + 1)]
        headers += ['centerline{}_x'.format(i) for i in range(1, len(self.centerline_x) + 1)]
        headers += ['centerline{}_y'.format(i) for i in range(1, len(self.centerline_y) + 1)]

        if self.intersection_x:
            headers += ['intersection{}_x'.format(i) for i in range(1, len(self.intersection_x) + 1)]
            headers += ['intersection{}_y'.format(i) for i in range(1, len(self.intersection_y) + 1)]
            headers += ['stopping_point{}_x'.format(i) for i in range(1, len(self.stopping_point_x) + 1)]
            headers += ['stopping_point{}_y'.format(i) for i in range(1, len(self.stopping_point_y) + 1)]

        # Append each header's respective values in following rows
        data = []
        for i in range(len(max(self.edges_x, key=len))):
            row = []
            for j in range(len(self.edges_x)):
                row.append(self.edges_x[j][i] if i < len(self.edges_x[j]) else '')
                row.append(self.edges_y[j][i] if i < len(self.edges_y[j]) else '')

            for j in range(len(self.centerline_x)):
                row.append(self.centerline_x[j][i] if i < len(self.centerline_x[j]) else '')
                row.append(self.centerline_y[j][i] if i < len(self.centerline_y[j]) else '')

            if self.intersection_x:
                for j in range(len(self.intersection_x)):
                    row.append(self.intersection_x[j][i] if i < len(self.intersection_x[j]) else '')
                    row.append(self.intersection_y[j][i] if i < len(self.intersection_y[j]) else '')

                for j in range(len(self.stopping_point_x)):
                    row.append(self.stopping_point_x[j][i] if i < len(self.stopping_point_x[j]) else '')
                    row.append(self.stopping_point_y[j][i] if i < len(self.stopping_point_y[j]) else '')

            data.append(row)

        # Check if savepath exists and write file to folder if yes
        if save_path is not None:
            path_exists = os.path.exists(save_path)
            if not path_exists:
                os.makedirs(save_path)
            file = os.path.join(save_path, filename)
        else:
            file = filename

        # Write data to csv file
        with open(file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(data)

    def check_track_size(self):
        """
        Checks if the track size is within the lab's size.
        Adjust the imported track used in the class initialization if an error persists.
        """

        # Check if track size is within parameters and print error and end program if not
        if (
            min(min(self.edges_x)) < -self.lab_x / 2
            or max(max(self.edges_x)) > self.lab_x / 2
            or min(min(self.edges_y)) < -self.lab_y / 2
            or max(max(self.edges_y)) > self.lab_y / 2
        ):
            print("Track size is larger than lab setting. Adjust track size.")
            exit()

    def animate_plot(self, timepoints, save=True):
        """
        Animates and plots the class data.

        :param save: True to save the animation as a GIF, False to display it on screen

        :param timepoints: Integer of frequency in Hz, that will dictate the time between 2 points. If timepoints=30, timestep is 0.033
        """

        # Set plot base
        fig, ax = plt.subplots()
        ax.set_xlim(min([item for sublist in self.edges_x for item in sublist]) - 1, max([item for sublist in self.edges_x for item in sublist]) + 1)
        ax.set_ylim(min([item for sublist in self.edges_y for item in sublist]) - 1, max([item for sublist in self.edges_y for item in sublist]) + 1)
        plt.gca().set_aspect('equal', adjustable='box')

        # Declare items on plot that update
        car_artists = []
        car_colors = ['red', 'blue', 'green']

        for i in range(len(self.centerline_x)):
            car_color = car_colors[i % len(car_colors)]  # Cycle through the available colors
            car = ax.plot([], [], color=car_color)[0]
            car_artists.append(car)

        time_label = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        timestep = 1 / timepoints

        # Animation
        def update(frame):
            num_centerlines = len(self.experiment_data_x)
            artists = []

            for i in range(num_centerlines):
                index = frame % len(self.experiment_data_x[i])

                if index + 1 >= len(self.experiment_data_x[i]):
                    angle = TrackUtils._calculate_angle(
                        self.experiment_data_x[i][index], self.experiment_data_y[i][index], self.experiment_data_x[i][0], self.experiment_data_y[i][0]
                    )
                else:
                    angle = TrackUtils._calculate_angle(
                        self.experiment_data_x[i][index], self.experiment_data_y[i][index], self.experiment_data_x[i][index + 1], self.experiment_data_y[i][index + 1]
                    )

                car_points = TrackUtils._rotated_rectangle(
                    self.experiment_data_x[i][index], self.experiment_data_y[i][index], self.rc_y, self.rc_x, angle
                )
                car_artists[i].set_data(*zip(*car_points))
                artists.append(car_artists[i])

            time_seconds = frame * timestep
            time_label.set_text('Time = {:.2f}s'.format(time_seconds))
            artists.append(time_label)

            return artists

        # Plot all static edges and centerlines of road
        for i in range(len(self.edges_x)):
            if i == 0 or i == len(self.edges_x) - 1:
                plt.plot(self.edges_x[i], self.edges_y[i], color='black', linewidth=3)
            else:
                plt.plot(self.edges_x[i], self.edges_y[i], color='black', linestyle='dashed')

        for i in range(len(self.centerline_x)):
            plt.plot(self.centerline_x[i], self.centerline_y[i], color='grey', linestyle='dashed')

        plt.title('Plot of Road')

        max_centerline_length = max(len(centerline) for centerline in self.experiment_data_x)
        # Animate plot
        ani = animation.FuncAnimation(fig, update, frames=max_centerline_length, interval=10, blit=True, cache_frame_data=False)

        if save:
            # Save plot
            writer = PillowWriter(fps=30)
            ani.save("road_animation.gif", writer=writer)
            plt.close()
        else:
            plt.show()

class TrackUtils:
    @staticmethod
    def _extract_coordinates(csv_file):
        # Extract coordinates from csv file and create dictionary
        data = {}
        with open(csv_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                for header, value in row.items():
                    if value:
                        if header not in data:
                            data[header] = []
                        data[header].append(float(value))

        return data

    @staticmethod
    def _calculate_angle(x1, y1, x2, y2):
        # Calculate angle change between current and previous points
        delta_x = x2 - x1
        delta_y = y2 - y1
        return math.degrees(math.atan2(delta_y, delta_x))

    @staticmethod
    def _rotated_rectangle(x, y, width, height, angle_deg):
        # Create rectangle with new changes for x, y, and angle
        half_width = width / 2
        half_height = height / 2

        angle_rad = math.radians(angle_deg)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        x1 = x - half_width * cos_theta + half_height * sin_theta
        y1 = y - half_width * sin_theta - half_height * cos_theta
        x2 = x + half_width * cos_theta + half_height * sin_theta
        y2 = y + half_width * sin_theta - half_height * cos_theta
        x3 = x + half_width * cos_theta - half_height * sin_theta
        y3 = y + half_width * sin_theta + half_height * cos_theta
        x4 = x - half_width * cos_theta - half_height * sin_theta
        y4 = y - half_width * sin_theta + half_height * cos_theta

        # Save coordinates for rectangle lines
        rectangle = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]

        return rectangle

    @staticmethod
    def _rotate_points(x_list, y_list, angle_rad):
        """
        Rotates a list of points given an angle in radians.

        :param x_list: list of x-coordinates
        :param y_list: list of y-coordinates
        :param angle_rad: rotation angle in radians
        :return: rotated x-coordinates and y-coordinates as a tuple
        """
        rotated_x = []
        rotated_y = []

        for x, y in zip(x_list, y_list):
            # Perform rotation transformation
            new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

            rotated_x.append(new_x)
            rotated_y.append(new_y)

        return rotated_x, rotated_y

if __name__ == "__main__":
    lab_x = 6.096  # 20 feet in meters
    lab_y = 9.144  # 30 feet in meters
    rc_x = 0.305  # 1 foot in meters
    rc_y = 0.610  # 2 feet in meters

    road_animation = TrackPlotter(lab_x, lab_y, rc_x, rc_y, 'tracks/oval_track.csv')
    #road_animation.check_track_size()
    road_animation.plot_trajec('oval_track_single_lane_pos2023-06-30_16_00_43.txt')
    #road_animation.add_translate([2.5, -1.5])
    #road_animation.rotate_track(45)
    #road_animation.save_track('translated_tracks', 'updated_coords.csv')
    road_animation.plot_track(show_plot=False)
    road_animation.animate_plot(timepoints=30, save=False)
