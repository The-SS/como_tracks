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

    def plot_trajec(self):
        # Plot trajectory of centerlines all on one plot
        num_centerlines = len(self.centerline_x)
        fig, ax = plt.subplots(num_centerlines, 3, figsize=(12, 4*num_centerlines))

        # Loop to plot all centerlines on plot
        for i in range(num_centerlines):
            # Plot trajectory
            ax[i, 0].plot(self.centerline_x[i], self.centerline_y[i], color='black')
            ax[i, 0].set_aspect('equal')
            ax[i, 0].set_title(f"Centerline {i+1} - Trajectory")
            ax[i, 0].set_xlabel("X")
            ax[i, 0].set_ylabel("Y")

            # Plot X coordinate
            ax[i, 1].plot(self.centerline_x[i])
            ax[i, 1].set_title(f"Centerline {i+1} - X Coordinate")

            # Plot Y coordinate
            ax[i, 2].plot(self.centerline_y[i])
            ax[i, 2].set_title(f"Centerline {i+1} - Y Coordinate")

        plt.tight_layout()
        plt.show()

    def add_translate(self, vector):
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
        # Check if track size is within paramaters and print error and end program if not
        if (
            min(min(self.edges_x)) < -self.lab_x / 2
            or max(max(self.edges_x)) > self.lab_x / 2
            or min(min(self.edges_y)) < -self.lab_y / 2
            or max(max(self.edges_y)) > self.lab_y / 2
        ):
            print("Track size is larger than lab setting. Adjust track size.")
            exit()

    def animate_plot(self, save):
        # Set plot base
        fig, ax = plt.subplots()
        ax.set_xlim(min([item for sublist in self.edges_x for item in sublist]) - 1, max([item for sublist in self.edges_x for item in sublist]) + 1)
        ax.set_ylim(min([item for sublist in self.edges_y for item in sublist]) - 1, max([item for sublist in self.edges_y for item in sublist]) + 1)
        plt.gca().set_aspect('equal', adjustable='box')

        # Declare items on plot that update
        car1 = ax.plot([], [], color='red')[0]
        car2 = ax.plot([], [], color='blue')[0]
        time_label = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)

        # Animation
        def update(frame):
            index_1 = frame % len(self.centerline_x[0])
            index_2 = frame % len(self.centerline_x[1])

            if index_1 + 1 >= len(self.centerline_x[0]):
                angle_1 = self._calculate_angle(self.centerline_x[0][index_1], self.centerline_y[0][index_1], self.centerline_x[0][0], self.centerline_y[0][0])
            else:
                angle_1 = self._calculate_angle(self.centerline_x[0][index_1], self.centerline_y[0][index_1], self.centerline_x[0][index_1 + 1], self.centerline_y[0][index_1 + 1])

            car1_points = self._rotated_rectangle(self.centerline_x[0][index_1], self.centerline_y[0][index_1], self.rc_y, self.rc_x, angle_1)
            car1.set_data(*zip(*car1_points))

            if index_2 + 1 >= len(self.centerline_x[1]):
                angle_2 = self._calculate_angle(self.centerline_x[1][index_2], self.centerline_y[1][index_2], self.centerline_x[1][0], self.centerline_y[1][0])
            else:
                angle_2 = self._calculate_angle(self.centerline_x[1][index_2], self.centerline_y[1][index_2], self.centerline_x[1][index_2 + 1], self.centerline_y[1][index_2 + 1])

            car2_points = self._rotated_rectangle(self.centerline_x[1][index_2], self.centerline_y[1][index_2], self.rc_y, self.rc_x, angle_2)
            car2.set_data(*zip(*car2_points))

            time_seconds = frame / 100
            time_label.set_text('Time = {:.2f}s'.format(time_seconds))

            return car1, car2, time_label

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
        ani = animation.FuncAnimation(fig, update, frames=3000, interval=1, blit=True, cache_frame_data=False)

        if (save):
            # Save plot
            writer = PillowWriter(fps=30)
            ani.save("road_animation.gif", writer=writer)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    lab_x = 20
    lab_y = 30
    rc_x = 1
    rc_y = 2

    road_animation = RoadAnimation(lab_x, lab_y, rc_x, rc_y, 'tracks/oval_track_two_centerline.csv')
    #road_animation.check_track_size()
    #road_animation.plot_trajec()
    #road_animation.add_translate([2.5, -1.5])
    #road_animation.rotate_track(45)
    #road_animation.save_track('translated_tracks', 'updated_coords.csv')
    road_animation.animate_plot(save=False)