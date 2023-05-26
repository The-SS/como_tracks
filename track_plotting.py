# https://www.geeksforgeeks.org/how-to-create-animations-in-python/
# https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Patch.html
# https://matplotlib.org/stable/api/animation_api.html


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv


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


def animate_plot(data):
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

    fig, ax = plt.subplots()

    ax.set_xlim(min(edge2_x + edge3_x) - 1, max(edge2_x + edge3_x) + 1)
    ax.set_ylim(min(edge2_x + edge3_y) - 1, max(edge2_x + edge3_y) + 1)

    line1, = ax.plot([], [], 'r-', lw=2)
    line2, = ax.plot([], [], 'b-', lw=2)

    max_length = 100

    def update(frame):
        start_index = max(0, frame - max_length)
        end_index = frame + 1
        line1.set_data(centerline1_x[start_index:end_index], centerline1_y[start_index:end_index])
        line2.set_data(centerline2_x[start_index:end_index], centerline2_y[start_index:end_index])
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=None, interval=0.5, blit=True)

    plt.plot(edge1_x, edge1_y, color='black', linewidth=3)
    plt.plot(edge3_x, edge3_y, color='black', linewidth=3)
    plt.plot(edge2_x, edge2_y, color='black', linestyle='dashed')
    plt.title('Plot of Road')

    plt.show()


if __name__ == "__main__":
    coordinates = extract_coordinates('tracks/oval_track_two_centerline.csv')
    animate_plot(coordinates)
