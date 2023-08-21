# COMO Tracks

Package that defines virtual tracks for the COMO platform at UTD 

## Package installation
The package can be installed using pip. Simply go to this package's root directory and run the following.
```commandline
pip install -e .
```


This README provides an overview of the TrackPlotter program, its functionality, and how to use it effectively.

## track_plotting
# Overview
The TrackPlotter program is designed to visualize and manipulate track data used in road simulations. It can read track coordinates from a CSV file, plot road edges and centerlines, perform translations and rotations of the track, and even animate a simulated car driving along the track.

# Prerequisites
- Python 3.x

# Required Libraries: 
- matplotlib

# How to Use
1. Ensure you have the necessary prerequisites installed.
   
2. Include the TrackPlotter class in your Python script.
   
3. Initialize an instance of the TrackPlotter class with the following parameters:
   - `lab_x`: Width of the lab in meters
   - `lab_y`: Height of the lab in meters
   - `rc_x`: Width of the rectangular car in meters
   - `rc_y`: Height of the rectangular car in meters
   - `csv_file`: Path to the CSV file containing the track coordinates.

4. Use the following methods to interact with the TrackPlotter instance:
   - `plot_trajec(txt_file)`: Plot collected data from a text file and compare it with simulated centerline data.
   - `plot_track()`: Plot static road edges and centerlines.
   - `add_translate(vector)`: Translate the entire track by a specified vector `[x, y]`.
   - `rotate_track(angle)`: Rotate the entire track by a specified angle in degrees.
   - `save_track(save_path, filename)`: Save the current track coordinates to a CSV file.
   - `check_track_size()`: Check if the track size is within the lab's size.
   - `animate_plot(timepoints, save)`: Animate and plot the class data. Set `save` to True to save the animation as a GIF, or False to display it on screen.


## two_lane_oval_track
# Description
This script generates road data for an oval-shaped track with two centerlines based on the given parameters. The track consists of two straight and parallel edges connected by two half-circular sides on the left and right. The total track length is defined by the sum of the lengths of the straight edges and the width is determined by the radius of the circular sides.

# Usage
Run the script to generate the oval track data. The generated data includes coordinates for the inner and outer edges, centerlines, and cumulative distances along the road.

# Parameters
- `L`: Length of straight edges (meters).
- `R`: Radius of circular sides (meters).
- `ds`: Discrete step size (meters).
- `verbose`: Print details about the track (default: True).
- `show`: Plot the track (default: True).
- `save`: Save the track data to a text file (default: True).
- `save_path`: Path to save the file (default: None).
- `filename`: Name of the saved file (default: 'track.csv').


## figure_eight_track
#  Description
This Python script generates a figure eight racetrack with two lanes defined as centerlines. The track is represented by coordinates for various elements such as borders, centerlines, stopping points, and intersections. It provides visualizations of the track and allows for saving the generated coordinates to a CSV file.

# Usage
This script defines a function named figure8_track, which generates the coordinates for the figure eight track. The track parameters (size, distance between centerline and border, discrete step size, etc.) can be adjusted within the function. The function returns the coordinates for various elements of the track.

# Parameters
- `I`: Size of Infinity symbol (meters)
- `B`: Distance between centerline and border on either side (meters)
- `ds`: Discrete step size (meters)
- `num_points`: Density for inner track creation (optional, default: 1000)
- `show`: Whether to display track visualizations (optional, default: True)
- `save`: Whether to save the generated track coordinates to a CSV file (optional, default: True)
- `save_path`: Path to save the CSV file (optional, default: None)
- `filename`: Name of the CSV file (optional, default: 'track.csv')