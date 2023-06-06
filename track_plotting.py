import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import csv
import math

def calculate_angle(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    return math.degrees(math.atan2(delta_y, delta_x))

def rotated_rectangle(x, y, width, height, angle_deg, color):
    half_width = width / 2
    half_height = height / 2

    rectangle = patches.Rectangle((x - half_width, y - half_height), width, height, fill=True, facecolor=color)
    angle_rad = math.radians(angle_deg)
    transform = transforms.Affine2D().rotate_around(x, y, angle_rad)
    rectangle.set_transform(transform + plt.gca().transData)

    return rectangle

# read csv to dictionary
def extract_coordinates(csv_file):
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

def animate_plot(coordinates, Lab_x, Lab_y, RC_x, RC_y):
    edge1_x = coordinates['edge1_x']
    edge1_y = coordinates['edge1_y']
    edge2_x = coordinates['edge2_x']
    edge2_y = coordinates['edge2_y']
    edge3_x = coordinates['edge3_x']
    edge3_y = coordinates['edge3_y']
    centerline1_x = coordinates['centerline1_x']
    centerline1_y = coordinates['centerline1_y']
    centerline2_x = coordinates['centerline2_x']
    centerline2_y = coordinates['centerline2_y']

    if min(edge3_x) < -Lab_x/2 or max(edge3_x) > Lab_x/2 or min(edge3_y) < -Lab_y/2 or max(edge3_y) > Lab_y/2:
        print("Track size is larger than lab setting. Adjust track size.")
        exit()

    fig, ax = plt.subplots()
    ax.set_xlim(min(edge2_x + edge3_x) - 1, max(edge2_x + edge3_x) + 1)
    ax.set_ylim(min(edge2_x + edge3_y) - 1, max(edge2_x + edge3_y) + 1)
    plt.gca().set_aspect('equal', adjustable='box')

    def update(frame):
        index_1 = frame % len(centerline1_x)
        index_2 = frame % len(centerline2_x)

        if index_1+1 >= len(centerline1_x):
            angle_1 = calculate_angle(centerline1_x[index_1], centerline1_y[index_1], centerline1_x[0], centerline1_y[0])
        else:
            angle_1 = calculate_angle(centerline1_x[index_1], centerline1_y[index_1], centerline1_x[index_1+1], centerline1_y[index_1+1])

        car1 = rotated_rectangle(centerline1_x[index_1], centerline1_y[index_1], RC_y, RC_x, angle_1, 'red')
        ax.add_patch(car1)

        if index_2+1 >= len(centerline2_x):
            angle_2 = calculate_angle(centerline2_x[index_2], centerline2_y[index_2], centerline2_x[0], centerline2_y[0])
        else:
            angle_2 = calculate_angle(centerline2_x[index_2], centerline2_y[index_2], centerline2_x[index_2+1], centerline2_y[index_2+1])

        car2 = rotated_rectangle(centerline2_x[index_2], centerline2_y[index_2], RC_y, RC_x, angle_2, 'blue')
        ax.add_patch(car2)

        return car1, car2

    plt.plot(edge1_x, edge1_y, color='black', linewidth=3)
    plt.plot(edge3_x, edge3_y, color='black', linewidth=3)
    plt.plot(edge2_x, edge2_y, color='black', linestyle='dashed')
    plt.plot(centerline1_x, centerline1_y, color='grey', linestyle='dashed')
    plt.plot(centerline2_x, centerline2_y, color='grey', linestyle='dashed')
    plt.title('Plot of Road')

    ani = animation.FuncAnimation(fig, update, frames=None, interval=1, blit=True, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    Lab_x = 30
    Lab_y = 20
    RC_x = 1
    RC_y = 2

    coordinates = extract_coordinates('tracks/oval_track_two_centerline.csv')
    animate_plot(coordinates, Lab_x, Lab_y, RC_x, RC_y)
