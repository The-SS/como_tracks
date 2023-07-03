import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import csv
import math

class RoadAnimation:
    def __init__(self, lab_x, lab_y, rc_x, rc_y, csv_file):
        # initialize variables and extract coordinates from csv file
        self.lab_x = lab_x
        self.lab_y = lab_y
        self.rc_x = rc_x
        self.rc_y = rc_y
        self.coordinates = self._extract_coordinates(csv_file)

        self.edges_x = []
        self.edges_y = []
        self.centerline_x = []
        self.centerline_y = []

        # Parse throught coordinates dictionary and append keys to corresponding lists
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

    def plot_trajec(self, txt_file):
        """
        Function takes text file of single centerline data with a format of:
        x, y, heading angle
        
        and displays collected data with simulated data to compare experimental variations

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

    def add_translate(self, vector):
        """
        Function will translate the track across a vector.
        Vector variable should should be a set of coordinates [x,y].

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
    
    def rotate_track(self, angle):
        """
        Function will rotate the track at an angle.
        Angle variable sent to track should be in degrees.

        """
        # Convert the angle from degrees to radians
        angle_rad = math.radians(angle)

        # Rotate the edges
        for i in range(len(self.edges_x)):
            self.edges_x[i], self.edges_y[i] = self._rotate_points(self.edges_x[i], self.edges_y[i], angle_rad)

        # Rotate the centerline
        for i in range(len(self.centerline_x)):
            self.centerline_x[i], self.centerline_y[i] = self._rotate_points(self.centerline_x[i], self.centerline_y[i], angle_rad)

    def _rotate_points(self, x_list, y_list, angle_rad):
        rotated_x = []
        rotated_y = []

        for x, y in zip(x_list, y_list):
            # Perform rotation transformation
            new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)

            rotated_x.append(new_x)
            rotated_y.append(new_y)

        return rotated_x, rotated_y

    def save_track(self, save_path, filename):
        """
        Function will save current coordinates whether they have been rotated, translated or neither.
        Save_path variable is the name directory of the folder to save to. 
        If given a name of "translated_tracks" the tracks will be saved in a folder called translated_tracks within the directory of the current file.

        Filename is the name of the saved file, make sure to end the filename with .csv

        """
        # Add all edge_x, edge_y, centerline_x, centerline_y numeration to header 
        headers = ['edge{}_x'.format(i) for i in range(1, len(self.edges_x) + 1)]
        headers += ['edge{}_y'.format(i) for i in range(1, len(self.edges_y) + 1)]
        headers += ['centerline{}_x'.format(i) for i in range(1, len(self.centerline_x) + 1)]
        headers += ['centerline{}_y'.format(i) for i in range(1, len(self.centerline_y) + 1)]

        # Append each headers respective values in following rows
        data = []
        for i in range(len(max(self.edges_x, key=len))):
            row = []
            for j in range(len(self.edges_x)):
                row.append(self.edges_x[j][i] if i < len(self.edges_x[j]) else '')
                row.append(self.edges_y[j][i] if i < len(self.edges_y[j]) else '')

            for j in range(len(self.centerline_x)):
                row.append(self.centerline_x[j][i] if i < len(self.centerline_x[j]) else '')
                row.append(self.centerline_y[j][i] if i < len(self.centerline_y[j]) else '')

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

    def _calculate_angle(self, x1, y1, x2, y2):
        # Calculate angle change between current and previous points
        delta_x = x2 - x1
        delta_y = y2 - y1
        return math.degrees(math.atan2(delta_y, delta_x))

    def _rotated_rectangle(self, x, y, width, height, angle_deg):
        # Create rectanle with with new changes for x, y and angle
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

    def _extract_coordinates(self, csv_file):
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

    def check_track_size(self):
        """
        Call function to check if the track is within the real labs size.
        Adjust your imported track used in the initialization of the class if error persists.
        
        """
        # Check if track size is within paramaters and print error and end program if not
        if (
            min(min(self.edges_x)) < -self.lab_x / 2
            or max(max(self.edges_x)) > self.lab_x / 2
            or min(min(self.edges_y)) < -self.lab_y / 2
            or max(max(self.edges_y)) > self.lab_y / 2
        ):
            print("Track size is larger than lab setting. Adjust track size.")
            exit()

    def animate_plot(self, save=True):
        """
        When function is called a animated plot of the class data is plotted and animated.
        If save variable is set to True a gif is downloaded.
        If save variable is set to False the plot is displayed on screen.

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

        # Animation
        def update(frame):
            num_centerlines = len(self.centerline_x)
            artists = []

            for i in range(num_centerlines):
                index = frame % len(self.centerline_x[i])

                if index + 1 >= len(self.centerline_x[i]):
                    angle = self._calculate_angle(self.centerline_x[i][index], self.centerline_y[i][index], self.centerline_x[i][0], self.centerline_y[i][0])
                else:
                    angle = self._calculate_angle(self.centerline_x[i][index], self.centerline_y[i][index], self.centerline_x[i][index + 1], self.centerline_y[i][index + 1])

                car_points = self._rotated_rectangle(self.centerline_x[i][index], self.centerline_y[i][index], self.rc_y, self.rc_x, angle)
                car_artists[i].set_data(*zip(*car_points))
                artists.append(car_artists[i])

            time_seconds = frame / 100
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

        # Animate plot
        ani = animation.FuncAnimation(fig, update, frames=None, interval=1, blit=True, cache_frame_data=False)

        if (save):
            # Save plot
            writer = PillowWriter(fps=30)
            ani.save("road_animation.gif", writer=writer)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    lab_x = 6.096 # 20 feet in meters
    lab_y = 9.144 # 30 feet in meters
    rc_x = 0.305  # 1 foot in meters
    rc_y = 0.610  # 2 feet in meters

    road_animation = RoadAnimation(lab_x, lab_y, rc_x, rc_y, 'tracks/oval_track.csv')
    road_animation.check_track_size()

    #road_animation.plot_trajec('oval_track_single_lane_pos2023-06-30_16_00_43.txt')
    #road_animation.add_translate([2.5, -1.5])
    #road_animation.rotate_track(45)
    #road_animation.save_track('translated_tracks', 'updated_coords.csv')

    road_animation.animate_plot(save=False)