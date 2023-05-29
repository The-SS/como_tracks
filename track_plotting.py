# https://www.geeksforgeeks.org/how-to-create-animations-in-python/
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
# https://matplotlib.org/stable/api/animation_api.html


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

def create_square(center_x, center_y, side_length):
    half_length = side_length / 2
    x = [center_x - half_length, center_x + half_length, center_x + half_length, center_x - half_length, center_x - half_length]
    y = [center_y - half_length, center_y - half_length, center_y + half_length, center_y + half_length, center_y - half_length]
    return x, y

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

    car1, = ax.plot([], [], 'r-', lw=2)
    car2, = ax.plot([], [], 'b-', lw=2)

    def update(frame):
        point_index = frame

        car1_index = point_index % len(centerline1_x)
        car2_index = point_index % len(centerline2_x)

        car1_x, car1_y = create_square(centerline1_x[car1_index], centerline1_y[car1_index], 1)
        car1.set_data(car1_x, car1_y)

        car2_x, car2_y = create_square(centerline2_x[car2_index], centerline2_y[car2_index], 1)
        car2.set_data(car2_x, car2_y)

        return car1, car2

    plt.plot(edge1_x, edge1_y, color='black', linewidth=3)
    plt.plot(edge3_x, edge3_y, color='black', linewidth=3)
    plt.plot(edge2_x, edge2_y, color='black', linestyle='dashed')
    plt.plot(centerline1_x, centerline1_y, color='grey', linestyle='dashed')
    plt.plot(centerline2_x, centerline2_y, color='grey', linestyle='dashed')
    plt.title('Plot of Road')

    
    ani = animation.FuncAnimation(fig, update, frames=None, interval=0.5, blit=True, cache_frame_data=False)

    plt.show()


if __name__ == "__main__":
    Lab_x = 30
    Lab_y = 20
    RC_x = 1
    RC_y = 2

    coordinates = extract_coordinates('tracks/oval_track_two_centerline.csv')
    animate_plot(coordinates, Lab_x, Lab_y, RC_x, RC_y)
