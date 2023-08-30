import os
import csv
from track_plotter import TrackPlotter


lab_x = 6.096
lab_y = 9.144
rc_x = 0.305
rc_y = 0.610


def load_single_lane_oval_track(fileName):
    track_data = TrackPlotter(lab_x, lab_y, rc_x, rc_y, fileName)
    return track_data.centerline_x[0], track_data.centerline_y[0]

def load_figure8_two_centerline_track(fileName):
    track_data = TrackPlotter(lab_x, lab_y, rc_x, rc_y, fileName)
    track_data.rotate_track(90)
    return track_data.centerline_x[1][:-1], track_data.centerline_y[1][:-1]
