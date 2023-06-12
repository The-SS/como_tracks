import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import csv
import math
from matplotlib.animation import PillowWriter

class RoadAnimation:
    def __init__(self, lab_x, lab_y, rc_x, rc_y, csv_file):
        self.lab_x = lab_x
        self.lab_y = lab_y
        self.rc_x = rc_x
        self.rc_y = rc_y
        self.coordinates = self.extract_coordinates(csv_file)
        self.edge1_x = self.coordinates['edge1_x']
        self.edge1_y = self.coordinates['edge1_y']
        self.edge2_x = self.coordinates['edge2_x']
        self.edge2_y = self.coordinates['edge2_y']
        self.edge3_x = self.coordinates['edge3_x']
        self.edge3_y = self.coordinates['edge3_y']
        self.centerline1_x = self.coordinates['centerline1_x']
        self.centerline1_y = self.coordinates['centerline1_y']
        self.centerline2_x = self.coordinates['centerline2_x']
        self.centerline2_y = self.coordinates['centerline2_y']

    def calculate_angle(self, x1, y1, x2, y2):
        delta_x = x2 - x1
        delta_y = y2 - y1
        return math.degrees(math.atan2(delta_y, delta_x))

    def rotated_rectangle(self, x, y, width, height, angle_deg, color):
        half_width = width / 2
        half_height = height / 2

        rectangle = patches.Rectangle((x - half_width, y - half_height), width, height, fill=True, facecolor=color)
        angle_rad = math.radians(angle_deg)
        transform = transforms.Affine2D().rotate_around(x, y, angle_rad)
        rectangle.set_transform(transform + plt.gca().transData)

        return rectangle

    def extract_coordinates(self, csv_file):
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
        if (min(self.edge3_x) < -self.lab_x/2 or max(self.edge3_x) > self.lab_x/2 or min(self.edge3_y) < -self.lab_y/2 or max(self.edge3_y) > self.lab_y/2):
            print("Track size is larger than lab setting. Adjust track size.")
            exit()

    def animate_plot(self):
        fig, ax = plt.subplots()
        ax.set_xlim(min(self.edge2_x + self.edge3_x) - 1, max(self.edge2_x + self.edge3_x) + 1)
        ax.set_ylim(min(self.edge2_x + self.edge3_y) - 1, max(self.edge2_x + self.edge3_y) + 1)
        plt.gca().set_aspect('equal', adjustable='box')

        def update(frame):
            index_1 = frame % len(self.centerline1_x)
            index_2 = frame % len(self.centerline2_x)

            if index_1+1 >= len(self.centerline1_x):
                angle_1 = self.calculate_angle(self.centerline1_x[index_1], self.centerline1_y[index_1], self.centerline1_x[0], self.centerline1_y[0])
            else:
                angle_1 = self.calculate_angle(self.centerline1_x[index_1], self.centerline1_y[index_1], self.centerline1_x[index_1+1], self.centerline1_y[index_1+1])

            car1 = self.rotated_rectangle(self.centerline1_x[index_1], self.centerline1_y[index_1], self.rc_y, self.rc_x, angle_1, 'red')
            ax.add_patch(car1)

            if index_2+1 >= len(self.centerline2_x):
                angle_2 = self.calculate_angle(self.centerline2_x[index_2], self.centerline2_y[index_2], self.centerline2_x[0], self.centerline2_y[0])
            else:
                angle_2 = self.calculate_angle(self.centerline2_x[index_2], self.centerline2_y[index_2], self.centerline2_x[index_2+1], self.centerline2_y[index_2+1])

            car2 = self.rotated_rectangle(self.centerline2_x[index_2], self.centerline2_y[index_2], self.rc_y, self.rc_x, angle_2, 'blue')
            ax.add_patch(car2)

            return car1, car2

        plt.plot(self.edge1_x, self.edge1_y, color='black', linewidth=3)
        plt.plot(self.edge3_x, self.edge3_y, color='black', linewidth=3)
        plt.plot(self.edge2_x, self.edge2_y, color='black', linestyle='dashed')
        plt.plot(self.centerline1_x, self.centerline1_y, color='grey', linestyle='dashed')
        plt.plot(self.centerline2_x, self.centerline2_y, color='grey', linestyle='dashed')
        plt.title('Plot of Road')

        ani = animation.FuncAnimation(fig, update, frames=None, interval=1, blit=True, cache_frame_data=False)

        plt.show()

if __name__ == "__main__":
    lab_x = 30
    lab_y = 20
    rc_x = 1
    rc_y = 2

    road_animation = RoadAnimation(lab_x, lab_y, rc_x, rc_y, 'tracks/oval_track_two_centerline.csv')
    road_animation.check_track_size()
    road_animation.animate_plot()
