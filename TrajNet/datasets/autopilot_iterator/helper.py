import math

import numpy as np

EARTH_RADIUS_METERS = 6371000.0


def radius_of_earth_at_latitude(B):
    B = math.radians(B)  # converting into radians
    a = 6378.137  # Radius at sea level at equator in Km
    b = 6356.752  # Radius at poles in Km
    c = (a**2 * math.cos(B)) ** 2
    d = (b**2 * math.sin(B)) ** 2
    e = (a * math.cos(B)) ** 2
    f = (b * math.sin(B)) ** 2
    R = math.sqrt((c + d) / (e + f))
    # return R * 1000.0 # Convert from kilometers to meters
    return R


def depth_color(val, heights, min_d=0, max_d=70):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    np.clip(
        val, 0, max_d, out=val
    )  # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


#
# def height_color(val, min_d=-15.44, max_d=2.778):
def height_color(depth, val, min_d=-2.357143, max_d=0.64285713):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val)
    return (((val - min_d) / (max_d - min_d)) * 255.0).astype(np.uint8)


def in_h_range_points(points, m, n, fov):
    """extract horizontal in-range points"""
    return np.logical_and(
        np.arctan2(n, m) > (-fov[1] * np.pi / 180),
        np.arctan2(n, m) < (-fov[0] * np.pi / 180),
    )


def in_v_range_points(points, m, n, fov):
    """extract vertical in-range points"""
    return np.logical_and(
        np.arctan2(n, m) < (fov[1] * np.pi / 180),
        np.arctan2(n, m) > (fov[0] * np.pi / 180),
    )


def fov_setting(points, x, y, z, dist, h_fov, v_fov):
    """filter points based on h,v FOV"""

    if (
        h_fov[1] == 180
        and h_fov[0] == -180
        and v_fov[1] == 2.0
        and v_fov[0] == -24.9
    ):
        return points

    if h_fov[1] == 180 and h_fov[0] == -180:
        return points[in_v_range_points(points, dist, z, v_fov)]
    elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
        return points[in_h_range_points(points, x, y, h_fov)]
    else:
        h_points = in_h_range_points(points, x, y, h_fov)
        v_points = in_v_range_points(points, dist, z, v_fov)
        return points[np.logical_and(h_points, v_points)]


def velo_points_filter(points, v_fov, h_fov, color_fn=depth_color):
    """extract points corresponding to FOV setting"""

    # Projecting to 2D
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    dist = np.sqrt(x**2 + y**2 + z**2)

    if h_fov[0] < -90:
        h_fov = (-90,) + h_fov[1:]
    if h_fov[1] > 90:
        h_fov = h_fov[:1] + (90,)

    x_lim = fov_setting(x, x, y, z, dist, h_fov, v_fov)[:, None]
    y_lim = fov_setting(y, x, y, z, dist, h_fov, v_fov)[:, None]
    z_lim = fov_setting(z, x, y, z, dist, h_fov, v_fov)[:, None]

    # Stack arrays in sequence horizontally
    xyz_ = np.hstack((x_lim, y_lim, z_lim))
    xyz_ = xyz_.T

    # stack (1,n) arrays filled with the number 1
    one_mat = np.full((1, xyz_.shape[1]), 1)
    xyz_ = np.concatenate((xyz_, one_mat), axis=0)

    # need dist info for points color
    dist_lim = fov_setting(dist, x, y, z, dist, h_fov, v_fov)
    color = color_fn(dist_lim.copy(), z.copy())

    return xyz_, color


def velo3d_2_camera2d_points(
    points,
    R_vc,
    T_vc,
    P_,
    v_fov=(-24.9, 2.0),
    h_fov=(-45, 45),
    color_fn=depth_color,
):
    """print velodyne 3D points corresponding to camera 2D image"""

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    # R_vc, T_vc = calib_velo2cam(vc_path)

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    # P_ = calib_cam2cam(cc_path, mode)

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV in the velodyne coordinates
    c_    - color value(HSV's Hue) corresponding to distance(m)

             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """
    xyz_v, c_ = velo_points_filter(points, v_fov, h_fov, color_fn=color_fn)

    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )

            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]
             [z_1 , z_2 , .. ]
    """
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::] / xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)

    """
    width = 1242
    height = 375
    w_range = in_range_points(ans[0], width)
    h_range = in_range_points(ans[1], height)

    ans_x = ans[0][np.logical_and(w_range,h_range)][:,None].T
    ans_y = ans[1][np.logical_and(w_range,h_range)][:,None].T
    c_ = c_[np.logical_and(w_range,h_range)]

    ans = np.vstack((ans_x, ans_y))
    """

    return ans, c_


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25**2).mean()
    d3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err**2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
