"""
Author: Hussein Jabak 
Email:  hjabak99@gmail.com
Date:   08/06/2023
"""

from track_plotter import *
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

class TrackAnimator(TrackPlotter):
    def __init__(self, lab_x, lab_y, rc_x, rc_y, csv_file):
        super().__init__(lab_x, lab_y, rc_x, rc_y, csv_file)        
            
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

if __name__ == "__main__":
    lab_x = 6.096  # 20 feet in meters
    lab_y = 9.144  # 30 feet in meters
    rc_x = 0.305  # 1 foot in meters
    rc_y = 0.610  # 2 feet in meters

    road_animation = TrackAnimator(lab_x, lab_y, rc_x, rc_y, 'tracks/otsl_track.csv')
    #road_animation.check_track_size()
    #road_animation.plot_trajec('oval_track_single_lane_pos2023-06-30_16_00_43.txt')
    #road_animation.add_translate([2.5, -1.5])
    #road_animation.rotate_track(45)
    #road_animation.save_track('translated_tracks', 'updated_coords.csv')
    #road_animation.plot_track(show_plot=False)
    road_animation.animate_plot(timepoints=30, save=False)
