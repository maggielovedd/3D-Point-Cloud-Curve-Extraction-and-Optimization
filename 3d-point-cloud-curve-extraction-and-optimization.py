import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spatial
from sklearn.neighbors import KDTree
from sklearn.model_selection import ParameterGrid
import vg
import time
from skopt import gp_minimize
from skopt.space import Real
from scipy.spatial import distance
from scipy.spatial import cKDTree
import signal

"""
3D Point Cloud Curve Extraction and Optimization

Purpose:
This code is designed to extract a representative curve (or a thin center line) from a noisy 3D point cloud, especially when the point cloud has discernible geometric patterns. 
Given a noisy point cloud that has an inherent structure or shape, the goal is to identify and trace a continuous curve that best captures the essence of that shape.

Key Features:
1. Synthetic Point Cloud Generation** - Allows the creation of synthetic 3D point clouds based on predefined shapes like curves, spirals, torus, and more, facilitating testing and demonstration.
2. Noise Simulation** - Introduces synthetic noise to generated points, simulating real-world noisy data scenarios.
3. Curve Thinning** - Implements algorithms to thin out the point cloud, emphasizing the central or most representative line.
4. Curve Sorting** - Ensures the extracted curve has points in a logical and continuous order.
5. Parameter Optimization** - Uses Bayesian optimization to determine the best processing parameters for curve extraction, ensuring optimal representation and continuity.
6. Visualization** - Provides real-time and post-process visualization of the original, thinned, and sorted curves.

Key Functions:
- `generate_3d_points()`: Generates 3D points for a given shape.
- `generate_sample_points()`: Adds noise to true point coordinates.
- `thin_line()`: Thins the point cloud to emphasize the curve or center line.
- `sort_points_on_regression_line()`: Sorts the thinned points to maintain a logical and continuous order.
- `sort_points()`: Combines the points sorted in both directions.
- `objective()`: Objective function for Bayesian optimization that drives the search for optimal parameters.
- `optimize_parameters()`: Uses Bayesian optimization to find the best parameters for curve extraction.

Execution:
On execution, the script will generate point clouds for various shapes, apply thinning and sorting processes, optimize parameters, and visualize the results, displaying the original point cloud alongside the extracted curve.

Notes:
- The code utilizes the `signal` library for timeout handling during optimization, which is compatible with UNIX platforms. For Windows users, a different approach for timeout handling may be needed.
- The curve extraction focuses on ensuring continuity and representation, making it suitable for point clouds where the underlying shape or pattern is discernible despite the noise.
"""

# == FUNCTIONS ========================================================================================================

def timeout_handler(signum, frame):
    """Handler for timeout."""
    raise TimeoutError("Function call timed out")


def objective(params):
    """
    Objective function for optimization.
    Args:
    - params (list): List of parameters to be optimized.
    Returns:
    - objective_value (float): Value of the objective function.
    """
    # Set the signal alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(3)  # Set timeout duration to 5 seconds

    try:
        point_cloud_thickness, sorted_point_distance, search_ratio = params

        # Round the parameters to desired precision
        point_cloud_thickness = round(point_cloud_thickness, 3)
        sorted_point_distance = round(sorted_point_distance, 3)
        search_ratio = round(search_ratio, 3)

        thinned_points, _ = thin_line(points, point_cloud_thickness)
        sorted_points = sort_points(thinned_points, _, sorted_point_distance, search_ratio)

        # If sorted_points has only one point, return a value representing its desirability
        if len(sorted_points) == 1:
            return -distance.euclidean(sorted_points[0], np.array([0, 0, 0]))
        
        # Calculate the total length of the sorted points
        total_length = sum(distance.euclidean(sorted_points[i], sorted_points[i+1]) for i in range(len(sorted_points)-1))
        
        # Calculate the average distance between consecutive points
        avg_distance = total_length / (len(sorted_points) - 1)
        
        # Calculate the penalty for discontinuities
        penalty_factor = 0.1
        penalty = sum(abs(distance.euclidean(sorted_points[i], sorted_points[i+1]) - avg_distance) for i in range(len(sorted_points)-1))
        penalty = penalty_factor * penalty

        # Calculate the reward based on the number of points in sorted_points
        # len(sorted_points) is somewhere around 40 given 1000 pt
        reward_factor = 0.01
        reward = len(sorted_points) * reward_factor

        # Combine the total length, the penalty, and the reward to form the objective
        objective_value = -total_length + penalty - reward

        # objective_value = -total_length
        objective_value = round(objective_value, 3)
        print(f"objective_value is {objective_value}")

        global number_of_iteration, real_time_ax
        number_of_iteration += 1

        # Visualization
        # Clear the previous plot
        real_time_ax.clear()

        # Plot the original, thinned, and sorted points
        real_time_ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o', color='#4e79a7', markersize=6)
        real_time_ax.plot(thinned_points.T[0], thinned_points.T[1], thinned_points.T[2], '+', color='#f28e2b', markersize=6)
        real_time_ax.scatter(sorted_points.T[0], sorted_points.T[1], sorted_points.T[2], c='#59a14f', marker='o', s=4**2, edgecolors='white')
        real_time_ax.plot(sorted_points.T[0], sorted_points.T[1], sorted_points.T[2], color='#bab0ac')

        # Display the iteration number and parameters as annotations on the plot
        real_time_ax.set_title(f"{shape}, Iteration {number_of_iteration}, Objective value: {objective_value}, \nThickness: {point_cloud_thickness:.3f}, Distance: {sorted_point_distance:.3f}, Ratio: {search_ratio:.3f}")
        real_time_ax.set_xlabel('X')
        real_time_ax.set_ylabel('Y')
        real_time_ax.set_zlabel('Z')

        # Update the figure
        plt.draw()
        plt.pause(0.001)  # Pause for 0.001 seconds

        # Reset the alarm
        signal.alarm(0)
        return objective_value

    except TimeoutError:
        print(f"Iteration {number_of_iteration} took too long. Skipping...")
        return 1e5  # Return a large value to indicate a bad objective


def optimize_parameters(points):
    """
    Optimize parameters for processing 3D point clouds using Bayesian optimization.
    
    Args:
    - points (np.array): The 3D point cloud to be processed.
    
    Returns:
    - list: The optimized parameters rounded to 3 decimal places.
    """

    # Define a global variable to keep track of the number of iterations
    global number_of_iteration
    number_of_iteration = 0

    # Use Bayesian optimization (Gaussian Process) to optimize the objective function
    res = gp_minimize(objective, space, n_calls=30, random_state=0)

    # Round the results to 3 decimal places and return
    return [round(x, 3) for x in res.x]


def thin_line(points, point_cloud_thickness=0.53, iterations=1, sample_points=0):
    """
    Thins a set of 3D points by projecting them onto their local regression line.
    
    Args:
    - points (np.array): Input points in the format [[x1, y1, z1], [x2, y2, z2], ...].
    - point_cloud_thickness (float): Radius for local neighborhood of points.
    - iterations (int): Number of iterations (currently unused).
    - sample_points (int): Number of points to sample from the beginning. If 0, use all points.
    
    Returns:
    - np.array: Transformed points.
    - list: Regression lines for each point.
    """
    
    if sample_points != 0:
        points = points[:sample_points]

    # Construct a KDTree for efficient nearest neighbor queries
    point_tree = cKDTree(points)

    new_points = []          # Transformed points
    regression_lines = []    # Regression lines for each point

    for point in points:
        # Find points within the specified radius
        points_in_radius = point_tree.data[point_tree.query_ball_point(point, point_cloud_thickness)]

        # Compute the mean of these points
        data_mean = points_in_radius.mean(axis=0)

        # Calculate the principal component (3D regression line) for these points
        _, _, vv = np.linalg.svd(points_in_radius - data_mean)
        linepts = vv[0] * np.mgrid[-1:1:2j][:, np.newaxis]
        linepts += data_mean
        regression_lines.append(list(linepts))

        # Project the original point onto the regression line
        ap = point - linepts[0]
        ab = linepts[1] - linepts[0]
        point_moved = linepts[0] + np.dot(ap, ab) / np.dot(ab, ab) * ab
        new_points.append(list(point_moved))

    return np.array(new_points), regression_lines


def sort_points_on_regression_line(points, regression_lines, index, sorted_point_distance, search_distance, direction=1, method='min_angle'):
    """
    Sorts points based on the specified method along the regression line.
    Various method can be applied here, and I found minimum angle gives the best result
    """
    sorted_points = []
    regression_line_prev = regression_lines[index][1] - regression_lines[index][0]
    point_tree = cKDTree(points)

    while True:
        v = regression_lines[index][1] - regression_lines[index][0]
        if np.dot(regression_line_prev, v) < 0:
            v = regression_lines[index][0] - regression_lines[index][1]
        regression_line_prev = v

        distR_point = points[index] + direction * (v / np.linalg.norm(v)) * sorted_point_distance
        points_in_radius = point_tree.data[point_tree.query_ball_point(distR_point, search_distance)]
        
        if len(points_in_radius) < 1:
            break

        # Minimum angle: choose the point that the line between this point and orginal point align with the regression line of original point  
        if method == 'min_angle':
            distR_point_vector = distR_point - points[index]
            angles = [vg.angle(distR_point_vector, x - points[index]) for x in points_in_radius]
            nearest_point = points_in_radius[np.argmin(angles)]
            index = np.where(points == nearest_point)[0][0]
        
        # Mean: choose the point that is nearest to the center of point in radius
        elif method == 'mean':
            mean_point = np.mean(points_in_radius, axis=0)
            index = (np.linalg.norm(points - mean_point, axis=1)).argmin()

        # Shortest Distance: Choose the point that is closest to the current point. 
        # This is a straightforward method and can be computed using the Euclidean distance.
        elif method == 'shortest_distance':
            distances = [np.linalg.norm(x - points[index]) for x in points_in_radius]
            nearest_point = points_in_radius[np.argmin(distances)]
            index = np.where(points == nearest_point)[0][0]

        # Maximum Dot Product: Instead of finding the smallest angle, you can find the point that has the maximum dot product with the regression line. 
        # This will give the point that is most aligned with the regression line.
        elif method == 'max_dot_product':
            dot_products = [np.dot(v, x - points[index]) for x in points_in_radius]
            nearest_point = points_in_radius[np.argmax(dot_products)]
            index = np.where(points == nearest_point)[0][0]

        # Density-Based: Choose the point that has the highest density of neighboring points within a certain radius. 
        # This can be useful if you want to prioritize areas with a higher concentration of points.
        elif method == 'density_based':
            densities = [len(point_tree.query_ball_point(x, search_distance)) for x in points_in_radius]
            nearest_point = points_in_radius[np.argmax(densities)]
            index = np.where(points == nearest_point)[0][0]

        # Curvature-Based: If the points represent a curve, you can compute the curvature at each point and 
        # choose the point with the highest curvature. This will prioritize points that are on sharper bends or turns.
        elif method == 'curvature_based':
            curvatures = []
            for x in points_in_radius:
                neighbors = point_tree.query_ball_point(x, search_distance)
                curvature = 0
                if len(neighbors) > 2:
                    a, b, c = points[neighbors[:3]]
                    curvature = np.linalg.norm(np.cross(b-a, c-a)) / (0.5 * np.linalg.norm(b-c))
                curvatures.append(curvature)
            nearest_point = points_in_radius[np.argmax(curvatures)]
            index = np.where(points == nearest_point)[0][0]
        
        sorted_points.append(points[index])
    
    return sorted_points


def sort_points(points, regression_lines, sorted_point_distance=0.2, search_ratio=1.2):
    """
    Sorts points along the regression line in both directions.
    """
    index = 0
    search_distance = sorted_point_distance / search_ratio

    sort_points_left = [points[index]] + sort_points_on_regression_line(points, regression_lines, index, sorted_point_distance, search_distance, direction=1)
    sort_points_right = sort_points_on_regression_line(points, regression_lines, index, sorted_point_distance, search_distance, direction=-1)

    return np.array(sort_points_left[::-1] + sort_points_right)


def generate_3d_points(shape, total_rad=10, z_factor=3, num_true_pts=1000):
    """
    Generate 3D points based on the specified shape.

    Parameters:
    - shape (str): The shape type for generating points.
    - total_rad (float): Total radians for the spiral. Default is 10, which means the spiral will rotate about 1.5 turns.
    - z_factor (float): Factor controlling the extension speed of the spiral on the z-axis. Smaller values make the spiral extend faster on the z-axis.
    - num_true_pts (int): Number of spiral points to generate.

    Returns:
    - x_true, y_true, z_true (np.array): Arrays containing the x, y, and z coordinates of the generated points.
    """

    s_true = np.linspace(0, total_rad, num_true_pts)
    
    if shape == "curve":
        x_true = np.cos(s_true)
        y_true = np.sin(s_true) * s_true
        z_true = s_true / z_factor

    elif shape == "line":
        A = np.array([0, 0, 0])  # Start point of the line
        B = np.array([z_factor, z_factor, z_factor])  # End point of the line
        t = np.linspace(0, 1, num_true_pts)
        x_true = A[0] + t * (B[0] - A[0])
        y_true = A[1] + t * (B[1] - A[1])
        z_true = A[2] + t * (B[2] - A[2])

    elif shape == "spiral":
        x_true = np.cos(s_true)
        y_true = np.sin(s_true)
        z_true = s_true / (2 * z_factor)

    elif shape == "waveform":
        x_true = s_true
        y_true = np.sin(s_true)
        z_true = np.cos(s_true)

    elif shape == "sinewave":
        x_true = s_true
        y_true = np.sin(3 * s_true)  # Triple frequency for y
        z_true = np.sin(2 * s_true)  # Double frequency for z

    elif shape == "spiral_cylinder":
        radius = 2
        x_true = radius * np.cos(s_true)
        y_true = radius * np.sin(s_true)
        z_true = s_true / z_factor

    elif shape == "spiral_cone":
        x_true = (total_rad - s_true) * np.cos(s_true)  # Radius decreases with s_true
        y_true = (total_rad - s_true) * np.sin(s_true)
        z_true = s_true / z_factor

    elif shape == "torus":
        R, r = 5, 2
        t_true = np.linspace(0, 2*np.pi, num_true_pts)
        x_true = (R + r * np.cos(s_true)) * np.cos(t_true)
        y_true = (R + r * np.cos(s_true)) * np.sin(t_true)
        z_true = r * np.sin(s_true)

    elif shape == "mobius":
        x_true = (1 + 0.5 * s_true * np.cos(0.5 * s_true)) * np.cos(s_true)
        y_true = (1 + 0.5 * s_true * np.cos(0.5 * s_true)) * np.sin(s_true)
        z_true = 0.5 * s_true * np.sin(0.5 * s_true)

    elif shape == "helicoid":
        t_true = np.linspace(-np.pi, np.pi, num_true_pts)  # adjust as needed
        x_true = s_true * np.cos(t_true)
        y_true = s_true * np.sin(t_true)
        z_true = t_true

    elif shape == "paraboloid":
        t_true = np.linspace(0, 2*np.pi, num_true_pts)
        x_true = np.sqrt(s_true) * np.cos(t_true)
        y_true = np.sqrt(s_true) * np.sin(t_true)
        z_true = s_true

    elif shape == "hyperboloid":
        t_true = np.linspace(0, 2*np.pi, num_true_pts)
        x_true = np.sqrt(s_true**2 + 1) * np.cos(t_true)
        y_true = np.sqrt(s_true**2 + 1) * np.sin(t_true)
        z_true = s_true

    elif shape == "ellipsoid":
        a, b, c = 5, 3, 2  # Semi-axes lengths, adjust as needed
        t_true = np.linspace(0, 2*np.pi, num_true_pts)
        phi = np.linspace(0, np.pi, num_true_pts)  # Second parameter for ellipsoid
        x_true = a * np.sin(phi) * np.cos(t_true)
        y_true = b * np.sin(phi) * np.sin(t_true)
        z_true = c * np.cos(phi)

    elif shape == "trefoil_knot":
        x_true = np.sin(s_true) + 2 * np.sin(2*s_true)
        y_true = np.cos(s_true) - 2 * np.cos(2*s_true)
        z_true = -np.sin(3*s_true)

    elif shape == "seashell":
        w = 0.5  # adjust as needed
        t_true = np.linspace(0, 2*np.pi, num_true_pts)
        x_true = (1 - s_true / (2*np.pi)) * np.cos(s_true) * (1 + np.cos(t_true)) + w * np.cos(s_true)
        y_true = (1 - s_true / (2*np.pi)) * np.sin(s_true) * (1 + np.cos(t_true)) + w * np.sin(s_true)
        z_true = w * np.sin(t_true) + 2 * s_true / (2*np.pi)
    
    # failed geometry
    # elif shape == "dini_surface":
    #     a, b = 1, 0.2  # Constants, adjust as needed
    #     t_true = np.linspace(0, 4*np.pi, num_true_pts)
    #     x_true = a * np.cos(s_true) * np.sin(t_true)
    #     y_true = a * np.sin(s_true) * np.sin(t_true)
    #     z_true = a * (np.cos(t_true) + np.log(np.tan(t_true / 2))) + b * s_true

    # elif shape == "circle":
    #     radius = z_factor
    #     x_true = radius * np.cos(s_true)
    #     y_true = radius * np.sin(s_true)
    #     z_true = np.full_like(s_true, 5)  # All points at z=5


    else:
        raise ValueError(f"Unknown shape: {shape}")
    
    return x_true, y_true, z_true


def generate_sample_points(x_true, y_true, z_true, noise=0.1):
    """
    Generate sample 3D points by adding noise to the true coordinates.

    Parameters:
    - x_true, y_true, z_true (np.array): True x, y, and z coordinates.
    - noise (float): Amount of noise to add to the true coordinates.

    Returns:
    - points (np.array): Sampled points in the format [(x1, y1, z1), (x2, y2, z2)...].
    """

    # Determine the number of sample points based on the length of x_true
    num_sample_pts = len(x_true)

    # Generate noisy sample points
    x_sample = x_true + noise * np.random.randn(num_sample_pts)
    y_sample = y_true + noise * np.random.randn(num_sample_pts)
    z_sample = z_true + noise * np.random.randn(num_sample_pts)

    # Stack the x, y, and z coordinates to form points
    points = np.vstack((x_sample, y_sample, z_sample)).T

    # Shuffle the points randomly
    np.random.shuffle(points)

    return points


# == MAIN ========================================================================================================

if __name__ == "__main__":
    """
    Main script to generate, optimize, and visualize 3D point clouds based on various shapes.
    """

    print("\nStarting...")

    # Create a figure for visualizing the results
    fig1 = plt.figure(figsize=(18, 10))

    # Set up a real-time visualization figure
    real_time_fig = plt.figure()
    real_time_ax = real_time_fig.add_subplot(111, projection='3d')

    # List of shapes to be processed
    shapes = ["curve", "line", "spiral", "waveform", "sinewave", \
              "spiral_cylinder", "spiral_cone", "torus", "mobius", \
              "helicoid", "paraboloid", "hyperboloid", "ellipsoid", "trefoil_knot", "seashell"]
    

    # Define the parameter search space for optimization
    space = [
        Real(0.4, 1, prior="uniform", name='point_cloud_thickness'),
        Real(0.05, 1, prior="uniform", name='sorted_point_distance'),
        Real(1.01, 4, prior="uniform", name='search_ratio')
    ]


    # Process each shape
    for i, shape in enumerate(shapes, start=1):

        start_time = time.perf_counter()

        # Set up the figure for real-time plotting
        plt.figure(real_time_fig.number)

        # Generate 3D points for the current shape
        x_true, y_true, z_true = generate_3d_points(shape)
        points = generate_sample_points(x_true, y_true, z_true)

        print(f"\nProcessing shape: {shape}")
        print(f"Number of points: {len(points)}")

        # Optimize parameters for the current shape
        best_params = optimize_parameters(points)

        # Extract the best parameters
        best_point_cloud_thickness, best_sorted_point_distance, best_search_ratio = best_params

        # Process the points using the best parameters
        thinned_points, regression_lines = thin_line(points, point_cloud_thickness=best_point_cloud_thickness)
        sorted_points = sort_points(thinned_points, regression_lines, sorted_point_distance=best_sorted_point_distance, search_ratio=best_search_ratio)

        # Print results for the current shape
        elapsed_time = time.perf_counter() - start_time
        print(f"Time taken for {shape}: {elapsed_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
 
        # Visualize the results
        ax = fig1.add_subplot(3, 5, i, projection='3d')
        title = ax.set_title(shape)
        title.set_position([.75, 0.9])

        # Plot the original, thinned, and sorted points
        ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o', color='#4e79a7', markersize=6)
        ax.plot(thinned_points.T[0], thinned_points.T[1], thinned_points.T[2], '+', color='#f28e2b', markersize=6)
        ax.scatter(sorted_points.T[0], sorted_points.T[1], sorted_points.T[2], c='#59a14f', marker='o', s=4**2, edgecolors='white')
        ax.plot(sorted_points.T[0], sorted_points.T[1], sorted_points.T[2], color='#bab0ac')

        # Display the best parameters and processing time in the subplot
        param_text = (f"Thickness: {best_point_cloud_thickness:.2f}\n"
                      f"Distance: {best_sorted_point_distance:.2f}\n"
                      f"Ratio: {best_search_ratio:.2f}\n"
                      f"Time: {elapsed_time:.2f} seconds")
        ax.text2D(-0.1, 0.95, param_text, transform=ax.transAxes, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
 
        # Update the figure
        plt.figure(fig1.number)
        plt.draw()
        plt.pause(0.001)

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()