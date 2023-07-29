import numpy as np
import matplotlib.pyplot as plt

#take normal to red border
#fix generate function

def generate_infinity_symbol_coordinates(a, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_infinity = a * np.cos(theta) / (1 + np.sin(theta)**2)
    y_infinity = a * np.cos(theta) * np.sin(theta) / (1 + np.sin(theta)**2)
    return x_infinity, y_infinity

def calculate_normals(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    magnitudes = np.sqrt(dx_dt**2 + dy_dt**2)

    # Calculate the normals
    nx = dy_dt / magnitudes
    ny = -dx_dt / magnitudes

    return nx, ny

def calculate_normals_flipped(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    magnitudes = np.sqrt(dx_dt**2 + dy_dt**2)

    # Calculate the normals
    nx = -dy_dt / magnitudes
    ny = dx_dt / magnitudes

    return nx, ny

# Choose the scaling factors
i_symbol_size = 2.0  # Adjust the value of 'a' to control the size of the infinity symbol
border_distance = 0.5  # Distance between the infinity symbol and the border
num_points = 1000  # Increase or decrease this value to get more or fewer points for a smoother or simpler curve


"""

GENERATE LEFT SIDE OF FIGURE EIGHT TRACK

"""


# Generate coordinates for the infinity symbol
x_inner_border, y_inner_border = generate_infinity_symbol_coordinates(i_symbol_size, num_points)

# Filter points with x > 0
mask = x_inner_border < 0
x_inner_border = x_inner_border[mask]
y_inner_border = y_inner_border[mask]

# Calculate the normal vectors to the infinity symbol
nx, ny = calculate_normals(x_inner_border, y_inner_border)
x_centerline1 = x_inner_border - border_distance * nx
y_centerline1 = y_inner_border - border_distance * ny

nx, ny = calculate_normals(x_centerline1, y_centerline1)
x_lane_border = x_centerline1 - border_distance * nx
y_lane_border = y_centerline1 - border_distance * ny

nx, ny = calculate_normals(x_lane_border, y_lane_border)
x_centerline2 = x_lane_border - border_distance * nx
y_centerline2 = y_lane_border - border_distance * ny

nx, ny = calculate_normals(x_centerline2, y_centerline2)
x_outer_border = x_centerline2 - border_distance * nx
y_outer_border = y_centerline2 - border_distance * ny



"""

GENERATE RIGHT SIDE OF FIGURE EIGHT TRACK

"""


# Generate flipped coordinates for the infinity symbol
x_inner_border_flip = -x_inner_border
y_inner_border_flip = y_inner_border

# Calculate the normal vectors to the flipped infinity symbol
nx, ny = calculate_normals_flipped(x_inner_border_flip, y_inner_border_flip)
x_centerline1_flip = x_inner_border_flip - border_distance * nx
y_centerline1_flip = y_inner_border_flip - border_distance * ny

nx, ny = calculate_normals_flipped(x_centerline1_flip, y_centerline1_flip)
x_lane_border_flip = x_centerline1_flip - border_distance * nx
y_lane_border_flip = y_centerline1_flip - border_distance * ny

nx, ny = calculate_normals_flipped(x_lane_border_flip, y_lane_border_flip)
x_centerline2_flip = x_lane_border_flip - border_distance * nx
y_centerline2_flip = y_lane_border_flip - border_distance * ny

nx, ny = calculate_normals_flipped(x_centerline2_flip, y_centerline2_flip)
x_outer_border_flip = x_centerline2_flip - border_distance * nx
y_outer_border_flip = y_centerline2_flip - border_distance * ny


"""

SHIFT TRACKS ACCORDINGLY


"""


# Get the shift value for the original lists
x_shift = x_outer_border[0]

# Move the original points by the shift value
x_inner_border_shifted = x_inner_border - x_shift
x_centerline1_shifted = x_centerline1 - x_shift
x_lane_border_shifted = x_lane_border - x_shift
x_centerline2_shifted = x_centerline2 - x_shift
x_outer_border_shifted = x_outer_border - x_shift

# Get the shift value for the flipped lists
x_shift_flip = x_outer_border_flip[0]

# Move the flipped points by the shift value
x_inner_border_flip_shifted = x_inner_border_flip - x_shift_flip
x_centerline1_flip_shifted = x_centerline1_flip - x_shift_flip
x_lane_border_flip_shifted = x_lane_border_flip - x_shift_flip
x_centerline2_flip_shifted = x_centerline2_flip - x_shift_flip
x_outer_border_flip_shifted = x_outer_border_flip - x_shift_flip

"""

CONNECT BOTH CURVES WITH STRAIGHT LINES TO FORM FIGURE EIGHT


"""

# Create straight lines with the same point density as the existing lists
num_points_straight_lines = len(x_centerline1)


# Generate the straight lines
x_straight_line1 = np.linspace(x_centerline1_shifted[-1], x_centerline2_flip_shifted[0], num_points_straight_lines)
y_straight_line1 = np.linspace(y_centerline1[-1], y_centerline2_flip[0], num_points_straight_lines)

x_straight_line2 = np.linspace(x_centerline2_flip_shifted[-1], x_centerline1_shifted[0], num_points_straight_lines)
y_straight_line2 = np.linspace(y_centerline2_flip[-1], y_centerline1[0], num_points_straight_lines)

# Concatenate the straight lines
x_concatenated_centerline1 = np.concatenate([x_centerline1_shifted, x_straight_line1, x_centerline2_flip_shifted, x_straight_line2])
y_concatenated_centerline1 = np.concatenate([y_centerline1, y_straight_line1, y_centerline2_flip, y_straight_line2])

# Generate the straight lines
x_straight_line1 = np.linspace(x_centerline2_shifted[-1], x_centerline1_flip_shifted[0], num_points_straight_lines)
y_straight_line1 = np.linspace(y_centerline2[-1], y_centerline1_flip[0], num_points_straight_lines)

x_straight_line2 = np.linspace(x_centerline1_flip_shifted[-1], x_centerline2_shifted[0], num_points_straight_lines)
y_straight_line2 = np.linspace(y_centerline1_flip[-1], y_centerline2[0], num_points_straight_lines)

# Concatenate the straight lines
x_concatenated_centerline2 = np.concatenate([x_centerline2_shifted, x_straight_line1, x_centerline1_flip_shifted, x_straight_line2])
y_concatenated_centerline2 = np.concatenate([y_centerline2, y_straight_line1, y_centerline1_flip, y_straight_line2])


# Generate the straight lines
x_straight_line1 = np.linspace(x_lane_border_shifted[-1], x_lane_border_flip_shifted[0], num_points_straight_lines)
y_straight_line1 = np.linspace(y_lane_border[-1], y_lane_border_flip[0], num_points_straight_lines)

x_straight_line2 = np.linspace(x_lane_border_flip_shifted[-1], x_lane_border_shifted[0], num_points_straight_lines)
y_straight_line2 = np.linspace(y_lane_border_flip[-1], y_lane_border[0], num_points_straight_lines)

# Concatenate the straight lines
x_concatenated_lane_border = np.concatenate([x_lane_border_shifted, x_straight_line1, x_lane_border_flip_shifted, x_straight_line2])
y_concatenated_lane_border = np.concatenate([y_lane_border, y_straight_line1, y_lane_border_flip, y_straight_line2])


"""

PLOT SEGMENTED TRACKS WITH STARTING POINTS FOR TROUBLESHOOTING


"""


# Plot both versions of the infinity symbol with border on the same graph
plt.figure(figsize=(10, 6))

# Original infinity symbol with border
plt.plot(x_inner_border_shifted, y_inner_border, label='Inner Border', color='tab:blue')
plt.plot(x_centerline1_shifted,  y_centerline1,  label='Centerline 1', color='tab:red')
plt.plot(x_lane_border_shifted,  y_lane_border,  label='Lane Border',  color='tab:blue')
plt.plot(x_centerline2_shifted,  y_centerline2,  label='Centerline 2', color='tab:red')
plt.plot(x_outer_border_shifted, y_outer_border, label='Outer Border', color='tab:blue')

# Plot the first point of each line with a different color and marker style
plt.scatter(x_inner_border_shifted[0], y_inner_border[0], color='tab:blue', marker='o', label='Start (Inner Border)')
plt.scatter(x_centerline1_shifted[0], y_centerline1[0], color='tab:green', marker='o', label='Start (Centerline 1)')
plt.scatter(x_lane_border_shifted[0], y_lane_border[0], color='tab:blue', marker='o', label='Start (Lane Border)')
plt.scatter(x_centerline2_shifted[0], y_centerline2[0], color='tab:red', marker='o', label='Start (Centerline 2)')
plt.scatter(x_outer_border_shifted[0], y_outer_border[0], color='tab:blue', marker='o', label='Start (Outer Border)')

# Flipped infinity symbol with border
plt.plot(x_inner_border_flip_shifted, y_inner_border_flip, label='Inner Border (Flipped)', color='tab:blue', linestyle='dashed')
plt.plot(x_centerline1_flip_shifted,  y_centerline1_flip,  label='Centerline 1 (Flipped)', color='tab:red', linestyle='dashed')
plt.plot(x_lane_border_flip_shifted,  y_lane_border_flip,  label='Lane Border (Flipped)',  color='tab:blue', linestyle='dashed')
plt.plot(x_centerline2_flip_shifted,  y_centerline2_flip,  label='Centerline 2 (Flipped)', color='tab:red', linestyle='dashed')
plt.plot(x_outer_border_flip_shifted, y_outer_border_flip, label='Outer Border (Flipped)', color='tab:blue', linestyle='dashed')

# Plot the first point of each line in the flipped version
plt.scatter(x_inner_border_flip_shifted[0], y_inner_border_flip[0], color='tab:blue', marker='o', linestyle='dashed', label='Start (Inner Border Flipped)')
plt.scatter(x_centerline1_flip_shifted[0], y_centerline1_flip[0], color='tab:purple', marker='o', linestyle='dashed', label='Start (Centerline 1 Flipped)')
plt.scatter(x_lane_border_flip_shifted[0], y_lane_border_flip[0], color='tab:blue', marker='o', linestyle='dashed', label='Start (Lane Border Flipped)')
plt.scatter(x_centerline2_flip_shifted[0], y_centerline2_flip[0], color='tab:red', marker='o', linestyle='dashed', label='Start (Centerline 2 Flipped)')
plt.scatter(x_outer_border_flip_shifted[0], y_outer_border_flip[0], color='tab:blue', marker='o', linestyle='dashed', label='Start (Outer Border Flipped)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Infinity Symbol with Border')
plt.axis('equal')
plt.legend()
plt.show()



# Plot both versions of the infinity symbol with border and the concatenated lines on the same graph
plt.figure(figsize=(10, 6))

# Original infinity symbol with border
plt.plot(x_inner_border_shifted, y_inner_border, label='Inner Border', color='tab:blue')
plt.plot(x_centerline1_shifted,  y_centerline1,  label='Centerline 1', color='tab:red')
plt.plot(x_lane_border_shifted,  y_lane_border,  label='Lane Border',  color='tab:blue')
plt.plot(x_centerline2_shifted,  y_centerline2,  label='Centerline 2', color='tab:red')
plt.plot(x_outer_border_shifted, y_outer_border, label='Outer Border', color='tab:blue')

# Flipped infinity symbol with border
plt.plot(x_inner_border_flip_shifted, y_inner_border_flip, label='Inner Border (Flipped)', color='tab:blue', linestyle='dashed')
plt.plot(x_centerline1_flip_shifted,  y_centerline1_flip,  label='Centerline 1 (Flipped)', color='tab:red', linestyle='dashed')
plt.plot(x_lane_border_flip_shifted,  y_lane_border_flip,  label='Lane Border (Flipped)',  color='tab:blue', linestyle='dashed')
plt.plot(x_centerline2_flip_shifted,  y_centerline2_flip,  label='Centerline 2 (Flipped)', color='tab:red', linestyle='dashed')
plt.plot(x_outer_border_flip_shifted, y_outer_border_flip, label='Outer Border (Flipped)', color='tab:blue', linestyle='dashed')

# Plot the concatenated lines as straight lines
plt.plot(x_concatenated_centerline1, y_concatenated_centerline1, label='Concatenated Lines', color='black', linestyle='dotted')
plt.plot(x_concatenated_centerline2, y_concatenated_centerline2, label='Concatenated Lines', color='black', linestyle='dotted')
plt.plot(x_concatenated_lane_border, y_concatenated_lane_border, label='Concatenated Lines', color='black', linestyle='dotted')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Infinity Symbol with Border and Concatenated Lines')
plt.axis('equal')
plt.legend()
plt.show()


"""
# Plot the original infinity symbol and its flipped version with the border
plt.figure(figsize=(12, 6))

# Original infinity symbol with border
plt.subplot(1, 2, 1)
plt.scatter(x_inner_border, y_inner_border, label='Inner Border', color='tab:blue')
plt.scatter(x_centerline1,  y_centerline1,  label='Centerline 1', color='tab:red')
plt.scatter(x_lane_border,  y_lane_border,  label='Lane Border',  color='tab:blue')
plt.scatter(x_centerline2,  y_centerline2,  label='Centerline 2', color='tab:red')
plt.scatter(x_outer_border, y_outer_border, label='Outer Border', color='tab:blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Original Infinity Symbol with Border')
plt.axis('equal')
plt.legend()

# Flipped infinity symbol with border
plt.subplot(1, 2, 2)
plt.scatter(x_inner_border_flip, y_inner_border_flip, label='Inner Border', color='tab:blue')
plt.scatter(x_centerline1_flip,  y_centerline1_flip,  label='Centerline 1', color='tab:red')
plt.scatter(x_lane_border_flip,  y_lane_border_flip,  label='Lane Border',  color='tab:blue')
plt.scatter(x_centerline2_flip,  y_centerline2_flip,  label='Centerline 2', color='tab:red')
plt.scatter(x_outer_border_flip, y_outer_border_flip, label='Outer Border', color='tab:blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Flipped Infinity Symbol with Border')
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()

"""


"""
# Calculate the border coordinates
x_border_outer = x_infinity + np.where(x_infinity > 0, border_distance * nx, -border_distance * nx)
y_border_outer = y_infinity + np.where(x_infinity > 0, border_distance * ny, -border_distance * ny)

# Iterate through the border coordinates and keep only points that are at least 0.5 away from ALL the infinity symbol points
border_coordinates_to_keep = []
for x_b, y_b in zip(x_border_outer, y_border_outer):
    distances = np.sqrt((x_b - x_infinity)**2 + (y_b - y_infinity)**2)
    if np.all(distances >= border_distance - 0.001):
        border_coordinates_to_keep.append((x_b, y_b))

# Separate the x and y coordinates again
if border_coordinates_to_keep:
    x_border_outer, y_border_outer = zip(*border_coordinates_to_keep)
else:
    x_border_outer, y_border_outer = [], []
"""


