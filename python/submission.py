"""
Homework 5
Submission Functions
"""

# Import necessary libraries
import numpy as np
import cv2 as cv
import helper as hlp
from scipy.signal import convolve2d

"""
Q3.1.1 Eight Point Algorithm
[I] points1: points in image 1 (Nx2 matrix)
    points2: points in image 2 (Nx2 matrix)
    max_dim: scalar value computed as max(H1, W1)
[O] F_matrix: the fundamental matrix (3x3 matrix)
"""
def normalize_points(points, max_dim):
    # Compute mean and standard deviation for normalization
    mean_vals = np.mean(points, axis=0)
    std_vals = np.std(points, axis=0)
    
    # Create the normalization transformation matrix
    norm_matrix = np.array([
        [1/std_vals[0], 0, -mean_vals[0]/std_vals[0]],
        [0, 1/std_vals[1], -mean_vals[1]/std_vals[1]],
        [0, 0, 1]
    ])
    
    # Convert points to homogeneous coordinates by appending a column of ones
    points_homog = np.hstack((points, np.ones((points.shape[0], 1))))
    # Apply the normalization matrix
    normalized = (norm_matrix @ points_homog.T).T
    return normalized[:, :2], norm_matrix

def eight_point(points1, points2, max_dim):
    # Ensure we have enough points (at least 8)
    assert max_dim >= 8
    num_points = points1.shape[0]

    # Normalize both sets of points
    norm_points1, T1_norm = normalize_points(points1, max_dim)
    norm_points2, T2_norm = normalize_points(points2, max_dim)

    # Construct the design matrix for the eight-point algorithm
    design_matrix = np.zeros((num_points, 9))
    for idx in range(num_points):
        x1, y1 = norm_points1[idx]
        x2, y2 = norm_points2[idx]
        design_matrix[idx] = [x1 * x2, x1 * y2, x1,
                              y1 * x2, y1 * y2, y1,
                              x2,       y2,      1]

    # Compute the SVD of the design matrix
    _, _, V_trans = np.linalg.svd(design_matrix)
    F_normalized = V_trans[-1].reshape(3, 3)

    # Enforce the rank-2 constraint on the fundamental matrix
    U_f, S_f, Vt_f = np.linalg.svd(F_normalized)
    S_f[2] = 0
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f

    # Optionally refine the fundamental matrix using a helper function
    F_refined = hlp.refineF(F_rank2, norm_points1, norm_points2)

    # Unscale the fundamental matrix back to the original coordinate system
    F_matrix = T2_norm.T @ F_refined @ T1_norm

    # Normalize F_matrix so that the bottom-right value is 1
    F_matrix /= F_matrix[2, 2]

    return F_matrix


"""
Q3.1.2 Epipolar Correspondences
[I] im1: image 1 (H1xW1 matrix)
    im2: image 2 (H2xW2 matrix)
    F_matrix: fundamental matrix from image 1 to image 2 (3x3 matrix)
    points1: points in image 1 (Nx2 matrix)
[O] points2: corresponding points in image 2 (Nx2 matrix)
"""
def compute_epipolar_lines(F_matrix, points1):
    # Append ones to convert points into homogeneous coordinates
    points1_homog = np.hstack((points1, np.ones((points1.shape[0], 1))))
    epi_lines = (F_matrix @ points1_homog.T).T
    return epi_lines

def epipolar_correspondences(im1, im2, F_matrix, points1):
    # Define the search window size for NCC computation
    win_size = 10
    half_window = win_size // 2

    # List to store corresponding points in the second image
    points2 = []
    
    # Convert images to grayscale
    gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    
    # Iterate over each point in the first image
    for pt in points1:
        point_x, point_y = int(pt[0]), int(pt[1])
        
        # Compute the epipolar line in image 2 using the fundamental matrix
        epi_line = F_matrix @ np.array([point_x, point_y, 1])
        a_coef, b_coef, c_coef = epi_line
        
        # Generate candidate matching points along the epipolar line
        height, width = gray2.shape
        candidate_list = []
        for x_candidate in range(max(0, point_x - 50), min(width, point_x + 50)):
            # Compute the corresponding y-coordinate on the epipolar line
            y_candidate = int((-a_coef * x_candidate - c_coef) / b_coef) if abs(b_coef) > 1e-6 else point_y
            if 0 <= y_candidate < height:
                candidate_list.append((x_candidate, y_candidate))
        
        # Extract a patch around the original point in image 1
        if (point_y - half_window < 0 or point_y + half_window >= height or 
            point_x - half_window < 0 or point_x + half_window >= width):
            points2.append((point_x, point_y))
            continue
        
        patch1 = gray1[point_y - half_window : point_y + half_window + 1,
                       point_x - half_window : point_x + half_window + 1]
        
        # Initialize variables for tracking the best match
        best_match_candidate = None
        best_ncc = -1  # NCC: higher value indicates a better match
        
        # Loop over candidate points in image 2 and compute NCC
        for (x_candidate, y_candidate) in candidate_list:
            if (y_candidate - half_window < 0 or y_candidate + half_window >= height or 
                x_candidate - half_window < 0 or x_candidate + half_window >= width):
                continue
            patch2 = gray2[y_candidate - half_window : y_candidate + half_window + 1,
                           x_candidate - half_window : x_candidate + half_window + 1]
            
            # Normalize both patches for cross-correlation
            norm_patch1 = (patch1 - np.mean(patch1)) / (np.std(patch1) + 1e-6)
            norm_patch2 = (patch2 - np.mean(patch2)) / (np.std(patch2) + 1e-6)
            ncc_val = np.sum(norm_patch1 * norm_patch2) / (win_size ** 2)
            
            if ncc_val > best_ncc:
                best_ncc = ncc_val
                best_match_candidate = (x_candidate, y_candidate)
        
        # Append the best match or fallback to the original point
        if best_match_candidate:
            points2.append(best_match_candidate)
        else:
            points2.append((point_x, point_y))
    
    return np.array(points2)


"""
Q3.1.3 Essential Matrix
[I] F_matrix: the fundamental matrix (3x3 matrix)
    cam_intrin1: camera matrix 1 (3x3 matrix)
    cam_intrin2: camera matrix 2 (3x3 matrix)
[O] E_matrix: the essential matrix (3x3 matrix)
"""
def essential_matrix(F_matrix, cam_intrin1, cam_intrin2):
    # Compute the essential matrix using the intrinsic parameters
    E_matrix = cam_intrin2.T @ F_matrix @ cam_intrin1
    U_e, S_e, Vt_e = np.linalg.svd(E_matrix)

    S_e[2] = 0  # Force the third singular value to zero (rank-2 constraint)
    E_matrix = U_e @ np.diag(S_e) @ Vt_e
    return E_matrix


"""
Q3.1.4 Triangulation
[I] proj_mat1: projection matrix for camera 1 (3x4 matrix)
    points1: image points from camera 1 (Nx2 matrix)
    proj_mat2: projection matrix for camera 2 (3x4 matrix)
    points2: image points from camera 2 (Nx2 matrix)
[O] points3D: computed 3D points in space (Nx3 matrix)
"""
def triangulate(proj_mat1, points1, proj_mat2, points2):
    num_pts = points1.shape[0]
    points3D_homo = np.zeros((num_pts, 4))

    # Process each point correspondence
    for idx in range(num_pts):
        x1, y1 = points1[idx]
        x2, y2 = points2[idx]

        # Build the system of equations: A * X = 0
        A_matrix = np.array([
            x1 * proj_mat1[2] - proj_mat1[0],
            y1 * proj_mat1[2] - proj_mat1[1],
            x2 * proj_mat2[2] - proj_mat2[0],
            y2 * proj_mat2[2] - proj_mat2[1]
        ])

        # Solve for the 3D point using SVD
        _, _, V_trans = np.linalg.svd(A_matrix)
        X_homo = V_trans[-1]  # Homogeneous solution

        # Normalize to make the last coordinate equal to 1
        points3D_homo[idx] = X_homo / X_homo[3]

    return points3D_homo


"""
Q3.2.1 Image Rectification
[I] cam_intrin1, cam_intrin2: camera matrices (3x3 matrix)
    rot1, rot2: rotation matrices (3x3 matrix)
    trans1, trans2: translation vectors (3x1 matrix)
[O] rect_M1, rect_M2: rectification matrices (3x3 matrix)
    rect_cam1, rect_cam2: rectified camera matrices (3x3 matrix)
    rect_rot1, rect_rot2: rectified rotation matrices (3x3 matrix)
    rect_trans1, rect_trans2: rectified translation vectors (3x1 matrix)
"""
def rectify_pair(cam_intrin1, cam_intrin2, rot1, rot2, trans1, trans2):
    # Calculate optical centers for both cameras
    center1 = -np.linalg.inv(cam_intrin1 @ rot1) @ (cam_intrin1 @ trans1)
    center2 = -np.linalg.inv(cam_intrin2 @ rot2) @ (cam_intrin2 @ trans2)

    # Compute the new rotation matrix using the baseline direction
    dir1 = (center1 - center2) / np.linalg.norm(center1 - center2)
    # Compute a perpendicular direction based on the original rotation's third row
    dir2 = np.cross(rot1[2, :], dir1) / np.linalg.norm(np.cross(rot1[2, :], dir1))
    dir3 = np.cross(dir2, dir1)

    new_rot = np.vstack((dir1, dir2, dir3))
    rect_rot1 = new_rot
    rect_rot2 = new_rot

    # Set rectified camera matrices (using the second camera's intrinsic as reference)
    rect_cam1 = cam_intrin2
    rect_cam2 = cam_intrin2

    # Calculate new translation vectors for rectified cameras
    rect_trans1 = -new_rot @ center1
    rect_trans2 = -new_rot @ center2

    # Compute the rectification (warp) matrices
    rect_M1 = (rect_cam1 @ rect_rot1) @ np.linalg.inv(cam_intrin1 @ rot1)
    rect_M2 = (rect_cam2 @ rect_rot2) @ np.linalg.inv(cam_intrin2 @ rot2)

    return rect_M1, rect_M2, rect_cam1, rect_cam2, rect_rot1, rect_rot2, rect_trans1, rect_trans2


"""
Q3.2.2 Disparity Map
[I] im_left: left image (H1xW1 matrix)
    im_right: right image (H2xW2 matrix)
    max_disparity: maximum disparity value (scalar)
    window_size: window size for block matching (scalar)
[O] disparity_map: computed disparity map (H1xW1 matrix)
"""
def get_disparity(im_left, im_right, max_disparity, window_size):
    height, width = im_left.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)
    half_window = window_size // 2

    # Pad images to handle border effects
    padded_left = np.pad(im_left, half_window, mode='constant', constant_values=0)
    padded_right = np.pad(im_right, half_window, mode='constant', constant_values=0)

    # Initialize a matrix to track the minimum sum of squared differences (SSD)
    min_ssd = np.full((height, width), np.inf)

    # Iterate over all possible disparity values
    for disp in range(max_disparity + 1):
        # Shift the right image horizontally by the disparity value
        shifted_right = np.roll(padded_right, -disp, axis=1)
        
        # Compute squared differences between the left image and the shifted right image
        ssd = (padded_left - shifted_right) ** 2
        
        # Convolve with a window to sum the squared differences locally
        window_kernel = np.ones((window_size, window_size))
        ssd_sum = convolve2d(ssd, window_kernel, mode='valid')
        
        # Update disparity map where a lower SSD is found
        mask = ssd_sum < min_ssd
        disparity_map[mask] = disp
        min_ssd[mask] = ssd_sum[mask]

    return disparity_map

"""
Q3.2.3 Depth Map
[I] disparity_map: disparity map (H1xW1 matrix)
    cam_intrin1, cam_intrin2: camera intrinsic matrices (3x3 matrix)
    rot1, rot2: rotation matrices (3x3 matrix)
    trans1, trans2: translation vectors (3x1 matrix)
[O] depth_map: computed depth map (H1xW1 matrix)
"""
def get_depth(disparity_map, cam_intrin1, cam_intrin2, rot1, rot2, trans1, trans2):
    # Compute optical centers for both cameras
    center1 = -np.linalg.inv(cam_intrin1 @ rot1) @ (cam_intrin1 @ trans1)
    center2 = -np.linalg.inv(cam_intrin2 @ rot2) @ (cam_intrin2 @ trans2)
    baseline = np.linalg.norm(center1 - center2)

    # Focal length from the left camera's intrinsic matrix (assumed at [0,0])
    focal_len = cam_intrin1[0, 0]

    # Prevent division by zero by replacing zeros with a small number
    safe_disp = np.where(disparity_map > 0, disparity_map, np.inf)

    # Calculate depth using the formula: depth = (baseline * focal_length) / disparity
    depth_map = (baseline * focal_len) / safe_disp

    # Set depth to zero where disparity was zero (no valid match)
    depth_map[disparity_map == 0] = 0

    return depth_map

"""
Q3.3.1 Camera Matrix Estimation
[I] image_pts: 2D points in the image (Nx2 matrix)
    world_pts: 3D points in space (Nx3 matrix)
[O] cam_proj_mat: estimated camera projection matrix (3x4 matrix)
"""
def estimate_pose(image_pts, world_pts):
    num_corr = image_pts.shape[0]
    eq_matrix = np.zeros((2 * num_corr, 12))
    
    # Build the linear system from the 2D-3D correspondences
    for idx in range(num_corr):
        X_val, Y_val, Z_val = world_pts[idx]
        u_val, v_val = image_pts[idx]
        
        eq_matrix[2 * idx] = [X_val, Y_val, Z_val, 1,
                                0, 0, 0, 0,
                                -u_val * X_val, -u_val * Y_val, -u_val * Z_val, -u_val]
        eq_matrix[2 * idx + 1] = [0, 0, 0, 0,
                                  X_val, Y_val, Z_val, 1,
                                  -v_val * X_val, -v_val * Y_val, -v_val * Z_val, -v_val]
    
    # Solve the system using SVD; the solution is the last row of V
    _, _, V_trans = np.linalg.svd(eq_matrix)
    cam_proj_mat = V_trans[-1].reshape(3, 4)
    
    return cam_proj_mat

"""
Q3.3.2 Camera Parameter Estimation
[I] cam_proj_mat: camera projection matrix (3x4 matrix)
[O] K_matrix: intrinsic parameters (3x3 matrix)
    R_matrix: rotation (3x3 matrix)
    t_vector: translation (3x1 vector)
"""
def estimate_params(cam_proj_mat):
    # Extract the left 3x3 part from the projection matrix
    M_matrix = cam_proj_mat[:, :3]
    
    # Perform an RQ decomposition via QR on the inverse to get intrinsic (K) and rotation (R)
    K_inv, R_inv = np.linalg.qr(np.linalg.inv(M_matrix))
    K_matrix = np.linalg.inv(K_inv)
    R_matrix = np.linalg.inv(R_inv)
    
    # Ensure the rotation has a positive determinant; if not, flip the signs
    if np.linalg.det(R_matrix) < 0:
        R_matrix = -R_matrix
        K_matrix = -K_matrix
    
    # Normalize the intrinsic matrix such that the bottom-right entry is 1
    K_matrix /= K_matrix[2, 2]
    
    # Compute the translation vector from the projection matrix
    t_vector = np.linalg.inv(K_matrix) @ cam_proj_mat[:, 3]
    
    return K_matrix, R_matrix, t_vector.reshape(-1, 1)

