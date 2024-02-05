import cv2
import numpy as np
import torch
from fastdtw import fastdtw

def trajectory_metrics(traj_gt, traj_pred):
    # Ensure the tensors are of the same shape
    assert traj_gt.shape == traj_pred.shape, "Shape mismatch between ground truth and predicted trajectories."

    # Compute the element-wise differences
    diff = traj_gt - traj_pred

    # Mean Absolute Error (MAE)
    mae = torch.mean(torch.abs(diff))

    # Mean Squared Error (MSE)
    mse = torch.mean(diff**2)

    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE)
    mape = torch.mean(torch.abs(diff) / (torch.abs(traj_gt) + 1e-8)) * 100

    # Cosine Similarity (assuming trajectories are vectors)
    cosine_similarity = torch.nn.functional.cosine_similarity(traj_gt.view(-1), traj_pred.view(-1), dim=0)

    # Dynamic Time Warping (DTW) - using the 'fastdtw' library
    dtw_distance, _ = fastdtw(traj_gt.cpu().numpy(), traj_pred.cpu().numpy())
    dtw_distance = torch.tensor(dtw_distance)

    return {
        'MAE': mae.item(),
        'MSE': mse.item(),
        'RMSE': rmse.item(),
        'MAPE': mape.item(),
        'Cosine Similarity': cosine_similarity.item(),
        'DTW Distance': dtw_distance.item()
    }


def apply_affine_transform_on_image_and_trajectory(
    frame_img,  # Image
    traj_gt,  # List of 3D points
    M_2D,  # 2D affine transform 3x3
    intrinsic_matrix,  # Camera Intrinsics 3x3
):
    K = np.hstack((intrinsic_matrix, np.zeros((3, 1))))
    N = np.linalg.pinv(K) @ M_2D @ K  # 3D Affine transform 4x4

    for point_index in range(traj_gt.shape[0]):
        p3d = np.array(
            [
                traj_gt[point_index][1] * 1,
                0.0,
                traj_gt[point_index][0] * -1,
                1.0,
            ]
        ).reshape(4, 1)
        p3d_new = N @ p3d
        traj_gt[point_index][0] = -p3d_new[2, 0]
        traj_gt[point_index][1] = p3d_new[0, 0]

    M = M_2D[:2, :3]
    frame_img = cv2.warpAffine(
        frame_img, M, (frame_img.shape[1], frame_img.shape[0])
    )

    return frame_img, traj_gt


def generate_morph(warp_y):
    M_2D = np.array(
        [
            [1.0, warp_y, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return M_2D


def affine_transform_2D(points, affine_transform):
    """
    Applies an affine transformation to a batched set of 2D points.
    :param points: torch.Tensor of shape (batch_size, num_points, 2)
    :param affine_transform: torch.Tensor of shape (batch_size, 3, 3)
    :return: transformed_points: torch.Tensor of shape (batch_size, num_points, 2)
    """
    # Append ones to the z-dimension of the points to create homogeneous coordinates
    batch_size, num_points, _ = points.size()
    points_hom = torch.cat(
        [
            points,
            torch.ones((batch_size, num_points, 1), device=points.device),
        ],
        dim=-1,
    )

    # Apply the affine transformation
    transformed_points_hom = torch.bmm(
        points_hom, affine_transform.transpose(1, 2)
    )

    # Convert back to 2D coordinates by dividing by the homogeneous coordinate
    transformed_points = (
        transformed_points_hom[:, :, :2] / transformed_points_hom[:, :, 2:]
    )

    return transformed_points


def absolute_to_relative(trajectory_absolute, inplace=True):
    """
    Converts a batched set of 2D points in absolute coordinates to relative coordinates.

    Args:
        trajectory_absolute (torch.Tensor): A tensor of shape (batch_size, num_points, 2) containing the absolute
        2D points for each trajectory in the batch.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_points, 2) containing the relative 2D points for each
        trajectory in the batch. The relative coordinates are specified as [dx, dy].
    """
    if inplace:
        return absolute_to_relative_inplace(trajectory_absolute)
    # Extract batch size and number of points
    batch_size, num_points, _ = trajectory_absolute.shape

    # Initialize the relative trajectory with zeros
    device = trajectory_absolute.device
    trajectory_relative = torch.zeros((batch_size, num_points, 2), device=device)

    # Loop over points in each trajectory
    for i in range(1, num_points):
        # Subtract the current absolute point from the previous absolute point
        trajectory_relative[:, i, :] = (
            trajectory_absolute[:, i, :] - trajectory_absolute[:, i - 1, :]
        )

    return trajectory_relative


def relative_to_absolute(trajectory_relative, inplace=True):
    """
    Converts a batched set of 2D points in relative coordinates to absolute coordinates.

    Args:
        trajectory_relative (torch.Tensor): A tensor of shape (batch_size, num_points, 2) containing the relative
        2D points for each trajectory in the batch. Each element is a 2D point specified as [dx, dy].

    Returns:
        torch.Tensor: A tensor of shape (batch_size, num_points, 2) containing the absolute 2D points for each
        trajectory in the batch. The first point in each trajectory is [0, 0].
    """
    if inplace:
        return relative_to_absolute_inplace(trajectory_relative)
    # Extract batch size and number of points
    batch_size, num_points, _ = trajectory_relative.shape

    # Initialize the absolute trajectory with zeros
    device = trajectory_relative.device
    trajectory = torch.zeros((batch_size, num_points, 2), device=device)

    # Loop over points in each trajectory
    for i in range(num_points):
        # Add the current relative point to the previous absolute point
        if i > 0:
            trajectory[:, i, :] = (
                trajectory[:, i - 1, :] + trajectory_relative[:, i, :]
            )

    return trajectory


def absolute_to_relative_inplace(trajectory_absolute):
    """
    Converts a batched set of 2D points in absolute coordinates to relative coordinates, in-place.

    Args:
        trajectory_absolute (torch.Tensor): A tensor of shape (batch_size, num_points, 2) containing the absolute
        2D points for each trajectory in the batch. The tensor will be modified in-place.
    """
    # Extract number of points
    _, num_points, _ = trajectory_absolute.shape

    # Loop over points in each trajectory, starting from the end and moving backward
    for i in range(num_points - 1, 0, -1):
        # Subtract the previous absolute point from the current absolute point
        trajectory_absolute[:, i, :] -= trajectory_absolute[:, i - 1, :]

    # Set the first point to [0, 0]
    trajectory_absolute[:, 0, :] = 0

    return trajectory_absolute


def relative_to_absolute_inplace(trajectory_relative):
    """
    Converts a batched set of 2D points in relative coordinates to absolute coordinates, in-place.

    Args:
        trajectory_relative (torch.Tensor): A tensor of shape (batch_size, num_points, 2) containing the relative
        2D points for each trajectory in the batch. The tensor will be modified in-place. The first point in each
        trajectory is [0, 0].
    """
    # Extract number of points
    _, num_points, _ = trajectory_relative.shape

    # Loop over points in each trajectory
    for i in range(1, num_points):
        # Add the current relative point to the previous absolute point
        trajectory_relative[:, i, :] += trajectory_relative[:, i - 1, :]
    return trajectory_relative


def plot_steering_traj(
    frame_center,
    trajectory,
    color=(255, 0, 0),
    intrinsic_matrix=None,
    DistCoef=None,
    offsets=[0.0, 1.5, 1.0],
    method="add_weighted",
):
    assert method in ("overlay", "mask", "add_weighted")
    if intrinsic_matrix is None:
        intrinsic_matrix = np.array(
            [
                [525.5030, 0, 333.4724],
                [0, 531.1660, 297.5747],
                [0, 0, 1.0],
            ]
        )
    if DistCoef is None:
        DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )
    h, w = frame_center.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        intrinsic_matrix, DistCoef, (w, h), 1, (w, h)
    )
    # frame['frame_center'] = cv2.undistort(frame['frame_center'], self.intrinsic_matrix, self.DistCoef, None, newcameramtx)
    # homo_cam_mat = np.hstack((newcameramtx, np.zeros((3,1))))
    homo_cam_mat = np.hstack((intrinsic_matrix, np.zeros((3, 1))))

    # rot = trajectory[0][:3,:3]
    # rot = np.eye(3,3)
    prev_point = None
    prev_point_3D = None
    rect_frame = np.zeros_like(frame_center)

    for trajectory_point in trajectory:
        p4d = np.ones((4, 1))
        p3d = np.array(
            [
                trajectory_point[1] * 1 - offsets[0],
                -offsets[1],
                trajectory_point[0] * -1 - offsets[2],
            ]
        ).reshape((3, 1))

        # If p3d is behind the camera, skip
        if p3d[2,0] > 0:
            continue

        # p3d = np.linalg.inv(rot) @ p3d
        p4d[:3, :] = p3d

        p2d = (homo_cam_mat) @ p4d
        if (
            p2d[2][0] != 0.0
            and not np.isnan(p2d).any()
            and not np.isinf(p2d).any()
        ):
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
            # frame_center = cv2.circle(frame_center, (px, py), 2, color, -1)
            if prev_point is not None:
                px_p, py_p = prev_point
                dist = ((px_p - px) ** 2 + (py_p - py) ** 2) ** 0.5
                if dist < 20:
                    rect_coords_3D = get_rect_coords_3D(p4d, prev_point_3D)
                    rect_coords = convert_3D_points_to_2D(
                        rect_coords_3D, homo_cam_mat
                    )
                    rect_frame = cv2.fillPoly(
                        rect_frame, pts=[rect_coords], color=color
                    )
                    # frame_center = cv2.addWeighted(frame_center, 1.0, rect_frame, 0.2, 0.0)
                    # frame_center = cv2.addWeighted(rect_frame, 0.2, frame_center, 1.0, 0.0)

                    frame_center = cv2.line(
                        frame_center, (px_p, py_p), (px, py), color, 2
                    )
                    # break

            prev_point = (px, py)
            prev_point_3D = p4d.copy()
        else:
            prev_point = None
            prev_point_3D = None

    if method == "mask":
        mask = np.logical_and(
            rect_frame[:, :, 0] == color[0],
            rect_frame[:, :, 1] == color[1],
            rect_frame[:, :, 2] == color[2],
        )
        frame_center[mask] = color
    elif method == "overlay":
        frame_center += (0.2 * rect_frame).astype(np.uint8)
    elif method == "add_weighted":
        cv2.addWeighted(frame_center, 0.8, rect_frame, 0.2, 0.0, frame_center)
    return frame_center


def steering_angle_list_2_traj(
    steering, velocity, time_deltas, steering_ratio, wheel_base, debug=False
):
    # steering: Steering wheel angle in degrees
    # velocity: velocity in meters/second
    # time_deltas: Time stamps in seconds

    # for time_index in range(1, self.trajectory_lookahead):
    for time_index in range(time_deltas.shape[0] - 1, 0, -1):
        time_deltas[time_index] = (
            time_deltas[time_index] - time_deltas[time_index - 1]
        )
    time_deltas[0] = 0.0

    # Front wheel angle in radians
    steering_angle = steering / steering_ratio * np.pi / 180.0
    distances = velocity * time_deltas

    if debug:
        print(
            "steering", np.min(steering), np.max(steering), np.mean(steering)
        )
        print(
            "velocity",
            np.min(velocity),
            np.max(velocity),
            np.mean(velocity),
            velocity.shape,
        )
        print(
            "time_deltas",
            np.min(time_deltas),
            np.max(time_deltas),
            np.mean(time_deltas),
            time_deltas.shape,
        )
        print(
            "distances", distances.shape, np.min(distances), np.max(distances)
        )

    thetas = np.zeros_like(time_deltas) * np.pi / 2.0
    for theta_i in range(1, thetas.shape[0]):
        thetas[theta_i] = thetas[theta_i - 1] + (
            velocity[theta_i]
            / wheel_base
            * np.tan(steering_angle[theta_i])
            * time_deltas[theta_i]
        )

    trajectory_x = distances * np.cos(thetas)
    trajectory_y = distances * np.sin(thetas)

    for traj_index in range(1, trajectory_x.shape[0]):
        trajectory_x[traj_index] = (
            trajectory_x[traj_index] + trajectory_x[traj_index - 1]
        )
        trajectory_y[traj_index] = (
            trajectory_y[traj_index] + trajectory_y[traj_index - 1]
        )

    trajectory_x = trajectory_x - trajectory_x[0]
    trajectory_y = trajectory_y - trajectory_y[0]

    trajectory = np.array([trajectory_x, trajectory_y]).T

    if debug:
        print("thetas", np.min(thetas), np.max(thetas), np.mean(thetas))
        print(
            "trajectory_x",
            np.min(trajectory_x),
            np.max(trajectory_x),
            np.mean(trajectory_x),
        )
        print(
            "trajectory_y",
            np.min(trajectory_y),
            np.max(trajectory_y),
            np.mean(trajectory_y),
        )
        print("trajectory", trajectory.shape)

    return trajectory


def plot_bev_trajectory(trajectory, frame_center, color=(0, 255, 0)):
    WIDTH, HEIGHT = frame_center.shape[1], frame_center.shape[0]
    traj_plot = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)

    X = -trajectory[:, 1]
    Z = trajectory[:, 0]

    X_min, X_max = -50.0, 50.0
    Z_min, Z_max = 0.0, 50.0
    X = (X - X_min) / (X_max - X_min)
    Z = (Z - Z_min) / (Z_max - Z_min)

    # X = (X - lb) / (ub - lb)
    # Z = (Z - lb) / (ub - lb)

    for traj_index in range(1, X.shape[0]):
        u = round(X[traj_index] * (HEIGHT - 1))
        v = round(Z[traj_index] * (WIDTH - 1))

        u_p = round(X[traj_index - 1] * (HEIGHT - 1))
        v_p = round(Z[traj_index - 1] * (WIDTH - 1))

        traj_plot = cv2.circle(traj_plot, (u, v), 5, color, -1)
        traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), color, 2)

    traj_plot = cv2.flip(traj_plot, 0)
    return traj_plot


def convert_3D_points_to_2D(points_3D, homo_cam_mat):
    points_2D = []
    for index in range(points_3D.shape[0]):
        p4d = points_3D[index]
        p2d = (homo_cam_mat) @ p4d
        px, py = 0, 0
        if p2d[2][0] != 0.0:
            px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])

        points_2D.append([px, py])

    return np.array(points_2D)


def get_rect_coords_3D(Pi, Pj, width=2.83972):
    x_i, y_i = Pi[0, 0], Pi[2, 0]
    x_j, y_j = Pj[0, 0], Pj[2, 0]
    points_2D = get_rect_coords(x_i, y_i, x_j, y_j, width)
    points_3D = []
    for index in range(points_2D.shape[0]):
        # point_2D = points_2D[index]
        point_3D = Pi.copy()
        point_3D[0, 0] = points_2D[index, 0]
        point_3D[2, 0] = points_2D[index, 1]

        points_3D.append(point_3D)

    return np.array(points_3D)



def trajectory_2_trajectory_rel(trajectory):
    trajectory_rel = trajectory.copy()
    for traj_index in range(trajectory.shape[0] - 1, 0, -1):
        trajectory_rel[traj_index] = (
            trajectory_rel[traj_index] - trajectory_rel[traj_index - 1]
        )
    trajectory_rel[0] = [0.0, 0.0]

    return trajectory_rel

def get_rect_coords(x_i, y_i, x_j, y_j, width=2.83972):
    Pi = np.array([x_i, y_i])
    Pj = np.array([x_j, y_j])
    height = np.linalg.norm(Pi - Pj)
    diagonal = (width**2 + height**2) ** 0.5
    D = diagonal / 2.0

    M = ((Pi + Pj) / 2.0).reshape((2,))
    theta = np.arctan2(Pi[1] - Pj[1], Pi[0] - Pj[0])
    theta += np.pi / 4.0
    # points = np.array([
    #     M + np.array([D*np.cos(theta+ 0*np.pi/2.0), D*np.sin(theta+ 0*np.pi/2.0)]),
    #     M + np.array([D*np.cos(theta+ 1*np.pi/2.0), D*np.sin(theta+ 1*np.pi/2.0)]),
    #     M + np.array([D*np.cos(theta+ 2*np.pi/2.0), D*np.sin(theta+ 2*np.pi/2.0)]),
    #     M + np.array([D*np.cos(theta+ 3*np.pi/2.0), D*np.sin(theta+ 3*np.pi/2.0)]),
    # ], dtype=np.int32)
    points = np.array(
        [
            M
            + np.array(
                [
                    D * np.sin(theta + 0 * np.pi / 2.0),
                    D * np.cos(theta + 0 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 1 * np.pi / 2.0),
                    D * np.cos(theta + 1 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 2 * np.pi / 2.0),
                    D * np.cos(theta + 2 * np.pi / 2.0),
                ]
            ),
            M
            + np.array(
                [
                    D * np.sin(theta + 3 * np.pi / 2.0),
                    D * np.cos(theta + 3 * np.pi / 2.0),
                ]
            ),
        ]
    )
    return points


def traverse_trajectory(traj, D):
    traj_interp = [traj[0]]
    dist = 0.0
    total_dist = 0.0
    for traj_i in range(1, traj.shape[0]):
        # traj_dist = np.linalg.norm(traj[traj_i, :] - traj[traj_i-1, :], ord=2)
        traj_dist = (
            (traj[traj_i, 0] - traj[traj_i - 1, 0]) ** 2
            + (traj[traj_i, 1] - traj[traj_i - 1, 1]) ** 2
        ) ** 0.5
        if dist + traj_dist > D:
            traj_interp.append(traj[traj_i - 1])
            dist = 0.0
        else:
            dist += traj_dist
            total_dist += traj_dist

    print("total_dist", total_dist)
    return np.array(traj_interp)


def trajectory_rel_2_trajectory(trajectory_rel):
    trajectory = trajectory_rel.copy()
    trajectory[0] = [0.0, 0.0]
    for traj_index in range(
        1, trajectory_rel.shape[0]
    ):  # TODO: Fix bug, traj_index-1 is negative for traj_index=0
        trajectory[traj_index] = (
            trajectory[traj_index - 1] + trajectory_rel[traj_index]
        )

    return trajectory

