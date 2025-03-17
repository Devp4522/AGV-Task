import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

# 1. Load temple images and correspondence data

# Load the point correspondences between images from file
corr_file = np.load("../data/some_corresp.npz")
img_points1 = corr_file["pts1"]
img_points2 = corr_file["pts2"]

# Read the two temple images
temple_image1 = io.imread("../data/im1.png")
temple_image2 = io.imread("../data/im2.png")

# 2. Compute the Fundamental Matrix using the eight-point algorithm

# Determine a scaling factor from the shape of the points array
scale_val = max(img_points1.shape)
# Compute the fundamental matrix with the eight_point function
F_matrix = sub.eight_point(img_points1, img_points2, scale_val)
# (Optional visualization call: hlp.displayEpipolarF(temple_image1, temple_image2, F_matrix))

# 3. Load temple coordinate points from file

# Load the temple coordinate points (from image 1) used for epipolar correspondence
temp_coords = np.load("../data/temple_coords.npz")
temple_coords_img1 = temp_coords["pts1"]

# 4. Compute corresponding points in image 2 using epipolar constraints

# Find the corresponding points in the second image
temple_coords_img2 = sub.epipolar_correspondences(temple_image1, temple_image2, F_matrix, temple_coords_img1)
# (Optional interactive matching: hlp.epipolarMatchGUI(temple_image1, temple_image2, F_matrix))

# 5. Compute the Essential Matrix and retrieve intrinsic camera parameters

# Load intrinsic parameters for both cameras
intrinsics = np.load("../data/intrinsics.npz")
cam1_intrin = intrinsics['K1']
cam2_intrin = intrinsics['K2']
# Compute the essential matrix using the fundamental matrix and intrinsics
E_matrix = sub.essential_matrix(F_matrix, cam1_intrin, cam2_intrin)
# print("Calculated essential matrix:", E_matrix)

# 6. Get the four possible camera projection matrices for camera 2 using camera2

# Retrieve possible camera matrices from helper function
all_possible_P2 = hlp.camera2(E_matrix)
proj_mats_cam2 = []
for idx in range(4):
    proj_mats_cam2.append(cam2_intrin @ all_possible_P2[:, :, idx])

# 7. Determine the correct camera projection matrix P2 by triangulating 3D points

# Define the projection matrix for the reference camera (Camera 1)
P1_matrix = cam1_intrin @ np.hstack((np.eye(3), np.zeros((3, 1))))

# Initialize variables for selecting the best P2 and associated 3D points
optimal_P2 = None
optimal_3D_points = None
max_valid_depth = 0

# Evaluate each candidate projection matrix for Camera 2
for proj_mat2 in proj_mats_cam2:
    # Triangulate the 3D points from the temple coordinate correspondences
    pts3D_homog = sub.triangulate(P1_matrix, temple_coords_img1, proj_mat2, temple_coords_img2)
    # Convert from homogeneous to Cartesian coordinates (ignoring the last coordinate)
    pts3D_cart = pts3D_homog[:, :3]
    
    # Compute depth values in both cameras (3rd row of projection result)
    depth_cam1 = (P1_matrix @ pts3D_homog.T)[2]
    depth_cam2 = (proj_mat2 @ pts3D_homog.T)[2]
    
    # Count the number of points that are in front of both cameras (positive depth)
    valid_depth_count = np.sum((depth_cam1 > 0) & (depth_cam2 > 0))
    
    # Update the best candidate if this projection yields more valid points
    if valid_depth_count > max_valid_depth:
        max_valid_depth = valid_depth_count
        optimal_P2 = proj_mat2
        optimal_3D_points = pts3D_cart

# Output the selected projection matrix and valid depth count
print("Selected Optimal Projection Matrix P2:", optimal_P2)
print("Max positive depth count:", max_valid_depth)

# 8. 3D Scatter Plot: Visualize the reconstructed 3D points

def plot_3D_points(points3D):
    """
    Generates a 3D scatter plot for a set of 3D points.
    
    Args:
        points3D (numpy array): Nx3 array containing the (X, Y, Z) coordinates.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract individual coordinates
    x_coords = points3D[:, 0]
    y_coords = points3D[:, 1]
    z_coords = points3D[:, 2]

    # Plot points with small blue markers
    ax.scatter(x_coords, y_coords, z_coords, c='b', marker='.', s=10)

    # Set axis labels
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")

    # Set view angle to match given visualization
    ax.view_init(elev=90, azim=90)

    plt.show()

# Plot the 3D points from the optimal solution
plot_3D_points(optimal_3D_points)

# 9. Save th

