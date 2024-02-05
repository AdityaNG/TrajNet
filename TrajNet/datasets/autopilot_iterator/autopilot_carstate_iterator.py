import math
import os
import pickle
import traceback

import cv2
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import Dataset

from .helper import *


class AutopilotCarStateIter(Dataset):
    def __init__(
        self,
        autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-03-08_13:22:28.134240",
        cam_left="rgb_1.mp4",
        cam_right="rgb_3.mp4",
        cam_center="rgb_2.mp4",
        car_state_csv="carState.csv",
        gps_csv="gps.csv",
        frame_skip=0,
        debug=False,
    ) -> None:
        self.autopilot_base_path = autopilot_base_path
        self.car_state_csv = os.path.join(
            self.autopilot_base_path, car_state_csv
        )
        self.gps_csv = os.path.join(self.autopilot_base_path, gps_csv)
        self.cam_left = os.path.join(self.autopilot_base_path, cam_left)
        self.cam_right = os.path.join(self.autopilot_base_path, cam_right)
        self.cam_center = os.path.join(self.autopilot_base_path, cam_center)
        self.frame_skip = frame_skip
        self.debug = debug

        assert os.path.isfile(self.car_state_csv), f"car_state_csv missing {self.car_state_csv}"
        assert os.path.isfile(self.gps_csv), f"gps_csv missing {self.gps_csv}"
        assert os.path.isfile(self.cam_left), f"cam_left missing {self.cam_left}"
        assert os.path.isfile(self.cam_right), f"cam_right missing {self.cam_right}"
        assert os.path.isfile(self.cam_center), f"cam_center missing {self.cam_center}"

        self.car_state = pd.read_csv(self.car_state_csv)
        self.gps = pd.read_csv(self.gps_csv)

        self.cam_left_cap = cv2.VideoCapture(self.cam_left)
        self.cam_right_cap = cv2.VideoCapture(self.cam_right)
        self.cam_center_cap = cv2.VideoCapture(self.cam_center)

        self.car_state = self.car_state.sort_values("timestamp")
        self.car_state_start_time = self.car_state["timestamp"].iloc[0]
        self.car_state_end_time = self.car_state["timestamp"].iloc[-1]

        self.gps = self.gps.sort_values("timestamp")
        self.gps_start_time = self.gps["timestamp"].iloc[0]
        self.gps_end_time = self.gps["timestamp"].iloc[-1]

        self.expected_duration = (
            self.car_state_end_time - self.car_state_start_time
        )

        self.car_state_fps = len(self.car_state) / self.expected_duration
        self.gps_fps = len(self.gps) / self.expected_duration

        if self.debug:
            print("car_state_fps", self.car_state_fps)
            print("gps_fps", self.gps_fps)

        assert self.car_state_start_time == self.gps_start_time
        assert self.car_state_end_time == self.gps_end_time
        assert self.car_state_fps == self.gps_fps

        if self.debug:
            print("cam_fps", self.cam_left_cap.get(cv2.CAP_PROP_FPS))
        self.cam_left_fps_true = self.car_state_fps
        self.cam_left_frame_count = int(
            self.cam_left_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.cam_left_duration = (
            self.cam_left_frame_count / self.cam_left_fps_true
        )
        self.cam_left_fps = self.cam_left_fps_true / (self.frame_skip + 1)

        # self.cam_right_fps = self.cam_right_cap.get(cv2.CAP_PROP_FPS)
        self.cam_right_fps_true = self.car_state_fps
        self.cam_right_frame_count = int(
            self.cam_right_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.cam_right_duration = (
            self.cam_right_frame_count / self.cam_right_fps_true
        )
        self.cam_right_fps = self.cam_right_fps_true / (self.frame_skip + 1)

        # self.cam_center_fps = self.cam_center_cap.get(cv2.CAP_PROP_FPS)
        self.cam_center_fps_true = self.car_state_fps
        self.cam_center_frame_count = int(
            self.cam_center_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )
        self.cam_center_duration = (
            self.cam_center_frame_count / self.cam_center_fps_true
        )
        self.cam_center_fps = self.cam_center_fps_true / (self.frame_skip + 1)

        if self.debug:
            print(
                "cam_left_duration, expected_duration",
                self.cam_left_duration,
                self.expected_duration,
            )
            print("cam_left_fps", self.cam_left_fps)
            print("cam_right_fps", self.cam_right_fps)
            print("cam_center_fps", self.cam_center_fps)

        assert (
            self.cam_left_fps == self.cam_right_fps
            and self.cam_right_fps == self.cam_center_fps
        )
        assert (
            self.cam_left_frame_count == self.cam_right_frame_count
            and self.cam_right_frame_count == self.cam_center_frame_count
        )
        assert (
            self.cam_left_duration == self.cam_right_duration
            and self.cam_right_duration == self.cam_center_duration
        )
        assert (
            abs(self.cam_left_duration - self.expected_duration) <= 1.0
        )  # Less than 1000 ms of drift

        self.duration = min(self.cam_left_duration, self.expected_duration)
        self.fps = self.car_state_fps

        self.size_true = int(
            self.duration * self.fps
        )  # Number of frames = FPS * number of seconds

        self.size = 0
        for _ in range(0, self.size_true, self.frame_skip + 1):
            self.size += 1

        self.old_frame_number = 0

    def __len__(self):
        return self.size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        data = self[self.index]
        self.index += 1
        return data

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        timestamp = self.car_state.loc[key][1]
        # time_from_start = timestamp - self.car_state_start_time
        # frame_number = round(time_from_start * self.fps)

        frame_number = key * (self.frame_skip + 1)

        # delta = abs(frame_number - self.old_frame_number)
        delta = frame_number - self.old_frame_number
        if frame_number >= self.old_frame_number and delta < 5 and delta > 0:
            for _ in range(delta - 1):
                ret, frame = self.cam_left_cap.read()
                ret, frame = self.cam_right_cap.read()
                ret, frame = self.cam_center_cap.read()
        else:
            self.cam_left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.cam_right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.cam_center_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            if self.debug:
                print("cap.set: ", delta)

        self.old_frame_number = frame_number

        ret_left, frame_left = self.cam_left_cap.read()
        ret_right, frame_right = self.cam_right_cap.read()
        ret_center, frame_center = self.cam_center_cap.read()
        if ret_left and ret_right and ret_center:
            frame_merged = np.concatenate(
                [
                    frame_left,
                    frame_center,
                    frame_right,
                ],
                axis=1,
            )
            car_state = self.car_state.iloc[[frame_number]]
            gps = self.gps.iloc[[frame_number]]
            return {
                "timestamp": timestamp,
                "frame_merged": frame_merged,
                "frame_left": frame_left,
                "frame_right": frame_right,
                "frame_center": frame_center,
                "car_state": car_state,
                "gps": gps,
            }

        raise IndexError(
            "Frame number not catured: ", frame_number, ", key=", key
        )


def get_autopilot_tree(autopilot_base_path):
    autopilot_dirs = []
    for root, dirs, files in os.walk(autopilot_base_path, topdown=False):
        for name in dirs:
            potential_autopilot_path = os.path.join(root, name)
            transforms_path = os.path.join(
                potential_autopilot_path, "transforms.json"
            )
            if os.path.isfile(transforms_path):
                autopilot_dirs.append(potential_autopilot_path)
    return autopilot_dirs


def get_all_autopilot_traj_datasets(autopilot_base_search, **kwargs):
    # Runs for 0m46.285s
    autopilot_dirs = get_autopilot_tree(autopilot_base_search)
    autopilot_datasets = []
    for autopilot_path in autopilot_dirs:
        autopilot_datasets.append(
            AutopilotIterTraj(autopilot_base_path=autopilot_path, **kwargs)
        )
    return autopilot_datasets


def get_all_autopilot_datasets(autopilot_base_search, **kwargs):
    autopilot_dirs = get_autopilot_tree(autopilot_base_search)
    autopilot_datasets = []
    for autopilot_path in autopilot_dirs:
        autopilot_datasets.append(
            AutopilotIter(autopilot_base_path=autopilot_path, **kwargs)
        )
    return autopilot_datasets


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR
    )
    return result


def plot_carstate_frame(
    frame_img,
    steering_pred=0.0,
    steering_gt=0.0,
    steering_img_path="media/steering.png",
):

    # frame_img = frame['frame_merged'].copy()

    frame_img_steering = np.zeros_like(frame_img)

    # car_state = frame['car_state']
    # steering_gt = car_state['steeringAngleDeg'].iloc[0]

    steering_dim = round(frame_img.shape[0] * 0.2)

    # Draw Steering GT
    steering_img = cv2.imread(steering_img_path)
    steering_img = rotate_image(steering_img, round(steering_gt))
    x_steering_start = round(frame_img_steering.shape[0] * 0.1)
    y_steering_start = (
        (frame_img_steering.shape[1] // 2) - (steering_dim // 2) - steering_dim
    )
    frame_img_steering[
        x_steering_start : x_steering_start + steering_dim,
        y_steering_start : y_steering_start + steering_dim,
    ] = cv2.resize(steering_img, (steering_dim, steering_dim))
    frame_img = cv2.addWeighted(frame_img, 1.0, frame_img_steering, 2.5, 0.0)

    # Draw Steering Pred
    steering_img = cv2.imread(steering_img_path)
    steering_img = rotate_image(steering_img, round(steering_pred))
    x_steering_start = round(frame_img_steering.shape[0] * 0.1)
    y_steering_start = (
        (frame_img_steering.shape[1] // 2) - (steering_dim // 2) + steering_dim
    )
    frame_img_steering[
        x_steering_start : x_steering_start + steering_dim,
        y_steering_start : y_steering_start + steering_dim,
    ] = cv2.resize(steering_img, (steering_dim, steering_dim))
    frame_img = cv2.addWeighted(frame_img, 1.0, frame_img_steering, -2.5, 0.0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    frame_img = cv2.putText(
        frame_img,
        "steering_gt: " + str(round(steering_gt, 2)),
        (50, 50),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    frame_img = cv2.putText(
        frame_img,
        "steer_pred: " + str(round(steering_pred, 2)),
        (50, 100),
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return frame_img


class AutopilotIterCarStateTraj(Dataset):
    def __init__(
        self,
        autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-03-08_13:22:28.134240",
        cam_left="rgb_1.mp4",
        cam_right="rgb_3.mp4",
        cam_center="rgb_2.mp4",
        car_state_csv="carState.csv",
        gps_csv="gps.csv",
        frame_skip=0,
        debug=False,
        load_cache=True,
        WIDTH=640,
        HEIGHT=480,
        pyslam_scale_factor=1.0,
        pyslam_num_features=8000,
        trajectory_lookahead=100,
        start_index=5,
    ) -> None:

        self.dataset = AutopilotCarStateIter(
            autopilot_base_path=autopilot_base_path,
            cam_left=cam_left,
            cam_right=cam_right,
            cam_center=cam_center,
            car_state_csv=car_state_csv,
            gps_csv=gps_csv,
            frame_skip=frame_skip,
            debug=debug,
        )
        self.load_cache = load_cache
        self.cache_path = os.path.join(autopilot_base_path, "cache_pyslam.pkl")

        self.trajectory_lookahead = trajectory_lookahead
        self.trajectory = []

        # 640x480 resolution
        # car_dataset/2023-03-02_16:07:47.876740_CALIB/rgb_2_calib/intrinsics.txt
        self.intrinsic_matrix = np.array(
            [
                [525.5030, 0, 333.4724],
                [0, 531.1660, 297.5747],
                [0, 0, 1.0],
            ]
        )
        self.DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )

        # self.DistCoef = np.array([
        #     0.0, 0.0,            # Tangential Distortion
        #     0.0, 0.0, 0.0    # Radial Distortion
        # ])

        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]
        self.camera_fps = self.dataset.cam_center_fps

        # assert 8.0 <= self.camera_fps <= 12.0, "FPS is not close to 10, choose a different value for frame_skip; camera_fps=" + str(self.camera_fps)

        print("self.camera_fps", self.camera_fps)
        # exit()

        self.pyslam_scale_factor = pyslam_scale_factor
        self.pyslam_num_features = pyslam_num_features

        if os.path.isfile(self.cache_path) and self.load_cache:
            print("Loading cache from:", self.cache_path)
            with open(self.cache_path, "rb") as cache_file:
                self.trajectory = pickle.load(cache_file)
        else:
            from pyslam.camera import PinholeCamera
            from pyslam.feature_tracker import feature_tracker_factory
            from pyslam.feature_tracker_configs import FeatureTrackerConfigs
            from pyslam.visual_odometry import VisualOdometry

            self.pyslam_cam = PinholeCamera(
                WIDTH * self.pyslam_scale_factor,
                HEIGHT * self.pyslam_scale_factor,
                self.fx * self.pyslam_scale_factor,
                self.fy * self.pyslam_scale_factor,
                self.cx * self.pyslam_scale_factor,
                self.cy * self.pyslam_scale_factor,
                self.DistCoef,
                self.camera_fps,
            )
            self.tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
            self.tracker_config["num_features"] = self.pyslam_num_features

            self.feature_tracker = feature_tracker_factory(
                **self.tracker_config
            )

            self.vo = VisualOdometry(
                self.pyslam_cam, None, self.feature_tracker
            )
            self.slam_img_id = 0

            # for frame_index in tqdm.tqdm(range(len(self.dataset))):
            for frame_index in tqdm.tqdm(
                range(start_index, 4 * self.trajectory_lookahead)
            ):

                frame = self.dataset[frame_index]
                frame_center = frame["frame_center"]
                transformation = np.eye(4, 4)

                try:
                    frame_center_resized = cv2.resize(
                        frame_center,
                        (0, 0),
                        fx=self.pyslam_scale_factor,
                        fy=self.pyslam_scale_factor,
                    )
                    frame_center_resized_greyscale = cv2.cvtColor(
                        frame_center_resized, cv2.COLOR_BGR2GRAY
                    )

                    # self.vo.track(frame_center_resized_greyscale, self.slam_img_id)
                    if not np.sum(frame_center_resized != (0, 0, 0)) == 0:
                        self.vo.track(frame_center_resized, self.slam_img_id)
                        self.slam_img_id += 1

                    if self.slam_img_id > 2:
                        x, y, z = self.vo.traj3d_est[-1]
                        rot = np.array(self.vo.cur_R, copy=True)
                        transformation[:3, :3] = rot
                        transformation[:3, 3] = [x, y, z]

                        cv2.imshow("frame_center_resized", self.vo.draw_img)
                        cv2.waitKey(1)

                except Exception as ex:
                    print(ex)
                    traceback.print_exc()
                    print("self.slam_img_id", self.slam_img_id)
                    exit()

                self.trajectory.append(transformation)

            print("Done generating, saving to:", self.cache_path)

            with open(self.cache_path, "wb") as cache_file:
                pickle.dump(self.trajectory, cache_file)

            print("Done generating cache")

    def __len__(self):
        return len(self.trajectory) - self.trajectory_lookahead

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        data = self[self.index]
        self.index += 1
        return data

    def __getitem__(self, frame_index):

        trajectory = self.trajectory[
            frame_index : frame_index + self.trajectory_lookahead
        ].copy()
        frame = self.dataset[frame_index]

        h, w = frame["frame_center"].shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.intrinsic_matrix, self.DistCoef, (w, h), 1, (w, h)
        )
        # frame['frame_center'] = cv2.undistort(frame['frame_center'], self.intrinsic_matrix, self.DistCoef, None, newcameramtx)
        # homo_cam_mat = np.hstack((newcameramtx, np.zeros((3,1))))
        homo_cam_mat = np.hstack((self.intrinsic_matrix, np.zeros((3, 1))))

        rot = trajectory[0][:3, :3]
        prev_point = None
        for transformation in trajectory:
            p4d = np.ones((4, 1))
            p3d = np.array(
                [
                    transformation[0, 3] - trajectory[0][0, 3],
                    transformation[1, 3] - trajectory[0][1, 3],
                    transformation[2, 3] - trajectory[0][2, 3],
                ]
            ).reshape((3, 1))
            p3d = np.linalg.inv(rot) @ p3d
            p4d[:3, :] = p3d

            p2d = (self.pyslam_scale_factor * homo_cam_mat) @ p4d
            if p2d[2][0] != 0.0:
                px, py = int(p2d[0][0] / p2d[2][0]), int(p2d[1][0] / p2d[2][0])
                frame["frame_center"] = cv2.circle(
                    frame["frame_center"], (px, py), 5, (0, 255, 0), -1
                )
                if prev_point is not None:
                    px_p, py_p = prev_point
                    frame["frame_center"] = cv2.line(
                        frame["frame_center"],
                        (px_p, py_p),
                        (px, py),
                        (0, 255, 0),
                        2,
                    )

                prev_point = (px, py)
        return {
            "frame_merged": frame["frame_merged"],
            "frame_left": frame["frame_left"],
            "frame_right": frame["frame_right"],
            "frame_center": frame["frame_center"],
            "car_state": frame["car_state"],
            "gps": frame["gps"],
            "trajectory": trajectory,
        }

from TrajNet.utils.trajectory import * 

class AutopilotIterCarStateTrajSteer(Dataset):
    def __init__(
        self,
        autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-03-08_13:22:28.134240",
        cam_left="rgb_1.mp4",
        cam_right="rgb_3.mp4",
        cam_center="rgb_2.mp4",
        car_state_csv="carState.csv",
        gps_csv="gps.csv",
        frame_skip=0,
        debug=False,
        trajectory_lookahead=250,
        start_index=5,
        wheel_base=2.83972,
        steering_ratio=13.27,
        chunk_size=1,
        **kwargs
    ) -> None:
        self.chunk_size = chunk_size
        self.wheel_base = wheel_base
        self.steering_ratio = steering_ratio
        self.trajectory_lookahead = trajectory_lookahead
        self.start_index = start_index
        self.debug = debug

        self.dataset = AutopilotCarStateIter(
            autopilot_base_path=autopilot_base_path,
            cam_left=cam_left,
            cam_right=cam_right,
            cam_center=cam_center,
            car_state_csv=car_state_csv,
            gps_csv=gps_csv,
            frame_skip=0,
            debug=debug,
        )
        self.frame_skip = frame_skip

        self.size_true = len(self.dataset)

        self.size = 0
        self.timestamps = []
        if self.debug:
            print("len(self.dataset.car_state)", len(self.dataset.car_state))

        for frame_index in range(
            self.start_index,
            self.size_true - self.trajectory_lookahead - self.chunk_size,
            self.frame_skip + 1,
        ):
            self.size += 1

        for frame_index_in in range(0, self.size, 1):
            frame_index = (
                frame_index_in * (self.frame_skip + 1)
            ) + self.start_index
            timestamp = self.dataset.car_state.loc[frame_index][1]
            self.timestamps.append(timestamp)

        if self.debug:
            print("min(self.timestamps)", min(self.timestamps))
            print("max(self.timestamps)", max(self.timestamps))

        # 640x480 resolution
        # car_dataset/2023-03-02_16:07:47.876740_CALIB/rgb_2_calib/intrinsics.txt
        self.intrinsic_matrix = np.array(
            [
                [525.5030, 0, 333.4724],
                [0, 531.1660, 297.5747],
                [0, 0, 1.0],
            ]
        )
        self.DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )

        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]
        self.camera_fps = self.dataset.cam_center_fps
        self.camera_time = 1.0 / self.camera_fps  # seconds

    def __len__(self):
        # return len(self.dataset) - self.trajectory_lookahead - self.start_index
        return self.size

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        data = self[self.index]
        self.index += 1
        return data

    def __getitem__(self, frame_index_in):
        if type(frame_index_in) == slice:
            return [
                self[i] for i in range(*frame_index_in.indices(self.__len__()))
            ]
        if frame_index_in >= self.__len__():
            raise IndexError

        # return [self.get_singular_frame(i) for i in range(frame_index_in, frame_index_in+self.chunk_size)]
        return [
            self.get_singular_frame(i)
            for i in range(
                frame_index_in + self.chunk_size, frame_index_in, -1
            )
        ]

    def get_singular_frame(self, frame_index_in):
        frame_index = (
            frame_index_in * (self.frame_skip + 1)
        ) + self.start_index

        frame = self.dataset[frame_index]

        # Steering wheel angle in degrees
        steering = np.array(
            self.dataset.car_state["steeringAngleDeg"]
            .iloc[frame_index : frame_index + self.trajectory_lookahead]
            .values.tolist()
        )
        steering = np.pad(
            steering,
            250 - steering.shape[0],
            mode="constant",
            constant_values=steering[-1],
        )
        steering = steering[0 : self.trajectory_lookahead]

        # velocity in meters/second
        velocity = np.array(
            self.dataset.car_state["vEgoRaw"]
            .iloc[frame_index : frame_index + self.trajectory_lookahead]
            .values.tolist()
        )
        velocity = np.pad(
            velocity,
            250 - velocity.shape[0],
            mode="constant",
            constant_values=0.0,
        )
        velocity = velocity[0 : self.trajectory_lookahead]

        # Time deltas in seconds
        time_deltas = np.array(
            self.dataset.car_state["timestamp"]
            .iloc[frame_index : frame_index + self.trajectory_lookahead]
            .values.tolist()
        )
        time_deltas = np.pad(
            time_deltas,
            250 - time_deltas.shape[0],
            mode="constant",
            constant_values=time_deltas[-1],
        )
        time_deltas = time_deltas[0 : self.trajectory_lookahead]

        unique_gps = self.dataset.gps[["lat", "lon", "bearingDeg"]]

        lats = np.array(
            unique_gps["lat"]
            .iloc[frame_index : frame_index + self.trajectory_lookahead]
            .values.tolist()
        )

        lons = np.array(
            unique_gps["lon"]
            .iloc[frame_index : frame_index + self.trajectory_lookahead]
            .values.tolist()
        )

        # Radians
        # bearingDegs = np.array(unique_gps['bearingDeg'].iloc[
        #     frame_index:frame_index+self.trajectory_lookahead
        # ].values.tolist())

        # if self.debug:
        #     print('bearingDegs', np.min(bearingDegs), np.max(bearingDegs), np.mean(bearingDegs), np.std(bearingDegs))

        waypoints = np.array([lats, lons]).T
        gps_position = waypoints[0]
        # heading in radians
        heading = compute_average_heading(
            waypoints[0 : min(10, len(waypoints))]
        )

        # gps_meters = gps_to_relative(waypoints, gps_position, bearingDegs[0])
        # gps_meters = gps_to_relative(waypoints, gps_position, 0.0)
        gps_meters = gps_to_relative(waypoints, gps_position, heading)

        if self.debug:
            print(
                "frame_index, trajectory_lookahead",
                frame_index,
                self.trajectory_lookahead,
            )
            print(
                "car_state['steeringAngleDeg']",
                self.dataset.car_state["steeringAngleDeg"],
            )
            print("steering.shape", steering.shape)
            print("velocity.shape", velocity.shape)
            print("time_deltas.shape", time_deltas.shape)

        assert steering.shape == (self.trajectory_lookahead,)
        assert velocity.shape == (self.trajectory_lookahead,)
        assert time_deltas.shape == (self.trajectory_lookahead,)

        # trajectory = steering_angle_list_2_traj(steering, velocity, time_deltas, self.steering_ratio, self.wheel_base, debug=self.debug)
        # trajectory = steering_angle_list_2_traj(steering, velocity, time_deltas, self.steering_ratio, self.wheel_base, debug=self.debug)
        trajectory = steering_angle_list_2_traj(
            steering.copy(),
            velocity.copy(),
            time_deltas.copy(),
            self.steering_ratio,
            self.wheel_base,
            debug=self.debug,
        )

        # trajectory_rel = trajectory_2_trajectory_rel(trajectory)
        trajectory_rel = trajectory_2_trajectory_rel(trajectory.copy())

        assert trajectory.shape == (self.trajectory_lookahead, 2)
        assert trajectory_rel.shape == (self.trajectory_lookahead, 2)

        if self.debug:
            print(
                "trajectory",
                trajectory.shape,
                np.min(trajectory),
                np.max(trajectory),
            )
            print(
                "trajectory_rel",
                trajectory_rel.shape,
                np.min(trajectory_rel),
                np.max(trajectory_rel),
            )

        return {
            "timestamp": frame["timestamp"],
            "frame_merged": frame["frame_merged"],
            "frame_left": frame["frame_left"],
            "frame_right": frame["frame_right"],
            "frame_center": frame["frame_center"],
            "car_state": frame["car_state"],
            "gps": frame["gps"],
            "steering": steering,
            "velocity": velocity,
            "trajectory": trajectory,
            "trajectory_rel": trajectory_rel,
            "gps_meters": gps_meters,
        }

import random

class AutopilotIterCarStateTrajSteer_torch(AutopilotIterCarStateTrajSteer):

    def __init__(self, transforms, *args, **kwargs):
        super(AutopilotIterCarStateTrajSteer_torch, self).__init__(*args, **kwargs)
        self.img_transform = transforms
        self.MAX_WARP_Y = 0.0
        if 'MAX_WARP_Y' in kwargs:
            self.MAX_WARP_Y = kwargs['MAX_WARP_Y']
        self.random_flip = True # If False, no random flip
        if 'random_flip' in kwargs:
            self.random_flip = kwargs['random_flip']

    def __len__(self,):
        new_len = super().__len__()
        if self.random_flip:
            new_len = new_len * 2
        return new_len

    def __getitem__(self, frame_index_original):
        original_len = super().__len__()
        frame_index = frame_index_original%original_len
        flip_flag = bool(frame_index_original//original_len)

        frames_list = super().__getitem__(frame_index)
        frame = frames_list[0]
        
        frame_merged_np = frame['frame_merged']
        frame_center_np = frame['frame_center']
        trajectory_rel = frame['trajectory_rel']

        if flip_flag:
            frame_merged_np = cv2.flip(frame_merged_np, 1)
            frame_center_np = cv2.flip(frame_center_np, 1)
            trajectory_rel[:,1] = - trajectory_rel[:,1]

        warp_y = (random.random() - 0.5) * self.MAX_WARP_Y
        M_2D = generate_morph(warp_y) # 2D affine transform 3x3

        frame_merged_np, trajectory_rel = apply_affine_transform_on_image_and_trajectory(
            frame_merged_np, trajectory_rel, M_2D, self.intrinsic_matrix
        )

        frame_merged = torch.tensor(frame_merged_np).unsqueeze(0)
        frame_center = torch.tensor(frame_center_np).unsqueeze(0)

        x = torch.tensor(self.img_transform({
            'image': frame_merged_np
        })['image']).unsqueeze(0)
        # x = x.permute((0,2,3,1))

        y = torch.tensor(trajectory_rel).unsqueeze(0)
        y = y.reshape((1, self.trajectory_lookahead*2))

        return [x, frame_merged, frame_center, y]


class AutopilotIterCarStateTrajSteer_torch_abs_velocity_2(AutopilotIterCarStateTrajSteer):

    def __init__(self, transforms, *args, **kwargs):
        super(AutopilotIterCarStateTrajSteer_torch_abs_velocity_2, self).__init__(*args, **kwargs)
        assert self.chunk_size==2, "chunk_size must be 2"
        self.img_transform = transforms
        self.MAX_WARP_Y = 0.0
        if 'MAX_WARP_Y' in kwargs:
            self.MAX_WARP_Y = kwargs['MAX_WARP_Y']
        self.random_flip = True # If False, no random flip
        if 'random_flip' in kwargs:
            self.random_flip = kwargs['random_flip']

    def __len__(self,):
        new_len = super().__len__()
        if self.random_flip:
            new_len = new_len * 2
        return new_len

    def __getitem__(self, frame_index_original):
        original_len = super().__len__()
        frame_index = frame_index_original%original_len
        flip_flag = bool(frame_index_original//original_len)

        frames_list = super().__getitem__(frame_index)
        frame = frames_list[0]
        frame_prev = frames_list[1]

        frame_merged_np = frame['frame_merged']
        frame_center_np = frame['frame_center']
        frame_merged_prev_np = frame_prev['frame_merged']
        frame_center_prev_np = frame_prev['frame_center']
        trajectory = frame['trajectory']
        velocity = frame['velocity']

        if flip_flag:
            frame_merged_np = cv2.flip(frame_merged_np, 1)
            frame_center_np = cv2.flip(frame_center_np, 1)
            frame_merged_prev_np = cv2.flip(frame_merged_prev_np, 1)
            frame_center_prev_np = cv2.flip(frame_center_prev_np, 1)
            trajectory[:,1] = - trajectory[:,1]

        warp_y = (random.random() - 0.5) * self.MAX_WARP_Y
        M_2D = generate_morph(warp_y) # 2D affine transform 3x3

        # print('trajectory', trajectory.shape)
        frame_merged_np, trajectory_transformed = apply_affine_transform_on_image_and_trajectory(
            frame_merged_np, trajectory.copy(), M_2D, self.intrinsic_matrix
        )

        frame_merged_prev_np, _ = apply_affine_transform_on_image_and_trajectory(
            frame_merged_prev_np, trajectory.copy(), M_2D, self.intrinsic_matrix
        )

        trajectory = trajectory_transformed

        frame_merged = torch.tensor(frame_merged_np).unsqueeze(0)
        frame_center = torch.tensor(frame_center_np).unsqueeze(0)

        frame_merged_torch = torch.tensor(self.img_transform({
            'image': frame_merged_np
        })['image']).unsqueeze(0)
        # frame_merged_torch = frame_merged_torch.permute((0,2,3,1))

        frame_merged_prev_torch = torch.tensor(self.img_transform({
            'image': frame_merged_prev_np
        })['image']).unsqueeze(0)
        # frame_merged_prev_torch = frame_merged_prev_torch.permute((0,2,3,1))

        trajectory_torch = torch.tensor(trajectory).unsqueeze(0)
        trajectory_torch = trajectory_torch.reshape((1, self.trajectory_lookahead, 2))
        
        velocity_torch = torch.tensor(velocity[0]).unsqueeze(0)
        velocity_torch = velocity_torch.reshape((1, 1))

        return [frame_merged_torch, frame_merged_prev_torch, frame_merged, frame_center, velocity_torch, trajectory_torch]

# class AutopilotIterCarStateTrajSteerTemporal(Dataset):

#     def __init__(self,
#         autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-03-08_13:22:28.134240",
#         cam_left='rgb_1.mp4',
#         cam_right='rgb_3.mp4',
#         cam_center='rgb_2.mp4',
#         car_state_csv='carState.csv',
#         gps_csv='gps.csv',
#         frame_skip=0,
#         debug=False,
#         trajectory_lookahead=250,
#         start_index=5,
#         wheel_base = 2.83972,
#         steering_ratio = 13.27,
#         chunk_size=2,
#         ) -> None:
#         self.traj_dataset = AutopilotIterCarStateTrajSteer(
#             autopilot_base_path=autopilot_base_path,
#             cam_left=cam_left,
#             cam_right=cam_right,
#             cam_center=cam_center,
#             car_state_csv=car_state_csv,
#             gps_csv=gps_csv,
#             frame_skip=frame_skip,
#             debug=debug,
#             trajectory_lookahead=trajectory_lookahead,
#             start_index=start_index,
#             wheel_base=wheel_base,
#             steering_ratio=steering_ratio,
#         )
#         self.chunk_size = chunk_size
#         self.size = len(self.traj_dataset) - self.chunk_size

#     def __len__(self):
#         # return len(self.dataset) - self.trajectory_lookahead - self.start_index
#         return self.size

#     def __iter__(self):
#         self.index = 0
#         return self

#     def __next__(self):
#         if self.index>=self.__len__():
#             raise StopIteration
#         data = self[self.index]
#         self.index += 1
#         return data

#     def __getitem__(self, frame_index):
#         return self.traj_dataset[frame_index:frame_index+self.chunk_size]

# def gps_to_relative(waypoints, position, heading):
#     """
#     Convert waypoints from GPS coordinates to relative cartesian coordinates in meters
#     with respect to the vehicle's current position and heading.

#     Args:
#         waypoints (np.array): List of waypoints in GPS coordinates of shape (N, 2)
#         position (np.array): Vehicle GPS coordinates of shape (2,)
#         heading (float): Vehicle GPS heading in radians
#     """
#     rel_waypoints = np.array(waypoints) - np.array(position)
#     cos_heading = np.cos(heading)
#     sin_heading = np.sin(heading)
#     rotation_matrix = np.array([[cos_heading, sin_heading], [-sin_heading, cos_heading]])
#     rel_waypoints = np.dot(rel_waypoints, rotation_matrix) * EARTH_RADIUS_METERS
#     # rel_waypoints = np.dot(rel_waypoints, rotation_matrix) * radius_of_earth_at_latitude(position[0]) * 10.0 # * 1000
#     # rel_waypoints = np.dot(rel_waypoints, rotation_matrix) * radius_of_earth_at_latitude(position[1]) * 10.0 # * 1000
#     # rel_waypoints[:1] = -rel_waypoints[:1]
#     return rel_waypoints


def gps_to_relative(waypoints, position, heading):
    """
    Convert waypoints from GPS coordinates to relative cartesian coordinates in meters
    with respect to the vehicle's current position and heading.

    Args:
        waypoints (np.array): List of waypoints in GPS coordinates of shape (N, 2)
        position (np.array): Vehicle GPS coordinates of shape (2,)
        heading (float): Vehicle GPS heading in radians

    Returns:
        np.array: List of waypoints in relative cartesian coordinates of shape (N, 2)
    """
    # Convert GPS coordinates to ECEF (Earth-Centered, Earth-Fixed) coordinates
    a = 6378137.0  # semi-major axis of the WGS84 ellipsoid in meters
    b = 6356752.3142  # semi-minor axis of the WGS84 ellipsoid in meters
    f = (a - b) / a  # flattening of the WGS84 ellipsoid
    e_sq = f * (2 - f)  # eccentricity squared of the WGS84 ellipsoid

    # waypoints_ecef = np.zeros_like(waypoints)
    waypoints_ecef = np.zeros((waypoints.shape[0], 3))
    for i, (lat, lon) in enumerate(waypoints):
        sin_lat = np.sin(np.deg2rad(lat))
        cos_lat = np.cos(np.deg2rad(lat))
        sin_lon = np.sin(np.deg2rad(lon))
        cos_lon = np.cos(np.deg2rad(lon))
        N = a / np.sqrt(1 - e_sq * sin_lat**2)
        x = (N + 0) * cos_lat * cos_lon
        y = (N + 0) * cos_lat * sin_lon
        z = (N * (1 - e_sq) + 0) * sin_lat
        waypoints_ecef[i] = np.array([x, y, z])

    position_ecef = np.zeros(3)
    lat, lon = position
    sin_lat = np.sin(np.deg2rad(lat))
    cos_lat = np.cos(np.deg2rad(lat))
    sin_lon = np.sin(np.deg2rad(lon))
    cos_lon = np.cos(np.deg2rad(lon))
    N = a / np.sqrt(1 - e_sq * sin_lat**2)
    x = (N + 0) * cos_lat * cos_lon
    y = (N + 0) * cos_lat * sin_lon
    z = (N * (1 - e_sq) + 0) * sin_lat
    position_ecef = np.array([x, y, z])

    # Convert ECEF coordinates to local tangent plane coordinates
    R = np.array(
        [
            [-np.sin(heading), np.cos(heading), 0],
            [-np.cos(heading), -np.sin(heading), 0],
            [0, 0, 1],
        ]
    )
    waypoints_tangent = np.dot(R, (waypoints_ecef - position_ecef).T).T

    # Convert local tangent plane coordinates to relative cartesian coordinates in meters
    waypoints_rel = waypoints_tangent[:, :2]
    waypoints_rel *= np.array([np.cos(heading), np.sin(heading)])

    return waypoints_rel


def compute_heading(P1, P2):
    """Calculate the heading direction in radians between two coordinates P1 and P2"""
    lat1, lon1 = P1
    lat2, lon2 = P2
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(dlon)
    heading = math.atan2(y, x)
    return heading


def compute_average_heading(points):
    """Calculate the average heading direction in radians between a list of N coordinates"""
    N = len(points)
    headings = []
    for i in range(N - 1):
        P1 = points[i]
        P2 = points[i + 1]
        heading = compute_heading(P1, P2)
        headings.append(heading)
    avg_heading = sum(headings) / len(headings)
    return avg_heading

from TrajNet.utils import img_transform
def get_carstate_dataset(
    dataset_base="/home/shared/car_dataset/car_dataset/",
    frame_skip=0,
    MAX_WARP_Y=0.0,
    random_flip=False,
    chunk_size=1,
    transforms=img_transform,
):
    import torch

    dataset = torch.utils.data.ConcatDataset(
        [
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-04-14_17:12:45.642824"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-27_16:55:45.578945"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-27_16:48:58.991218"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-07_17:32:11.996225"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-07_17:26:46.807798"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-07_17:20:35.564092"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-08_13:22:28.134240"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-08_13:14:18.350532"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-08_12:45:11.632434"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-08_12:27:06.251047"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
            AutopilotIterCarStateTrajSteer_torch(
                autopilot_base_path=os.path.join(
                    dataset_base, "2023-03-08_11:09:50.635084"
                ),
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                frame_skip=frame_skip,
                MAX_WARP_Y=MAX_WARP_Y,
                random_flip=random_flip,
                chunk_size=chunk_size,
                transforms=transforms,
            ),
        ]
    )
    return dataset


# def get_all_autopilot_traj_datasets(autopilot_base_search, **kwargs):
#     autopilot_dirs = ..dataautopilot_iterator.get_autopilot_tree(autopilot_base_search)
#     autopilot_datasets = []
#     for autopilot_path in autopilot_dirs:
#         autopilot_datasets.append(
#             AutopilotIterTraj(
#                 autopilot_base_path=autopilot_path,
#                 **kwargs
#             )
#         )
#     return autopilot_datasets


def main():
    dataset = AutopilotIterCarStateTrajSteer(
        # autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-03-07_17:20:35.564092",
        # autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-04-18_22:31:05.159478",
        autopilot_base_path="/home/aditya/Datasets/car_dataset/2023-04-18_22:27:36.166407",
        trajectory_lookahead=250,
        frame_skip=1,
        debug=True,
        chunk_size=2,
    )

    print("Done loading dataset")

    HEIGHT = 640
    WIDTH = 480

    # for frame_index in range(len(dataset)):
    #     frame_list = dataset[frame_index:frame_index+3]
    for frame_list in dataset:
        frame = frame_list[1]
        traj_plot = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)

        timestamp = frame["timestamp"]
        frame_center = frame["frame_center"]
        frame_merged = frame["frame_merged"]
        steering = frame["steering"]
        velocity = frame["velocity"]
        trajectory = frame["trajectory"]
        trajectory_rel = frame["trajectory_rel"]
        gps_meters = frame["gps_meters"]

        print("=" * 20)
        print("timestamp", timestamp)
        print("=" * 20)

        assert trajectory.shape == (dataset.trajectory_lookahead, 2)
        assert trajectory_rel.shape == (dataset.trajectory_lookahead, 2)

        X = -trajectory[:, 1]
        Z = trajectory[:, 0]

        X_min, X_max = np.min(X), np.max(X)
        Z_min, Z_max = np.min(Z), np.max(Z)

        lb = min(X_min, Z_min)
        ub = max(X_max, Z_max)

        print("lb, ub", lb, ub)

        # lb, ub = -15.0, 15.0
        lb, ub = -10.0, 80.0
        print("X", np.min(X), np.max(X), X.shape)
        print("Z", np.min(Z), np.max(Z), Z.shape)

        X_min, X_max = -50.0, 50.0
        Z_min, Z_max = 0.0, 50.0
        X = (X - X_min) / (X_max - X_min)
        Z = (Z - Z_min) / (Z_max - Z_min)

        # X = (X - lb) / (ub - lb)
        # Z = (Z - lb) / (ub - lb)

        # for traj_index in range(1, X.shape[0]):
        #     u = round(X[traj_index] * (HEIGHT - 1))
        #     v = round(Z[traj_index] * (WIDTH - 1))

        #     u_p = round(X[traj_index-1] * (HEIGHT - 1))
        #     v_p = round(Z[traj_index-1] * (WIDTH - 1))

        #     traj_plot = cv2.circle(traj_plot, (u, v), 5, (0,255,0), -1)
        #     traj_plot = cv2.line(traj_plot, (u_p, v_p), (u, v), (0,255,0), 2)

        print("gps_meters.shape", gps_meters.shape)
        traj_plot = plot_bev_trajectory(
            trajectory, frame_center, color=(0, 255, 0)
        )
        traj_plot_dps = plot_bev_trajectory(
            gps_meters, frame_center, color=(0, 255, 0)
        )
        traj_plot = cv2.addWeighted(traj_plot, 0.5, traj_plot_dps, 0.5, 0.0)
        traverse_trajectory(trajectory, 2.2 * 0.1)

        assert np.isclose(
            trajectory_rel_2_trajectory(trajectory_rel), trajectory
        ).all()

        cv2.imshow("traj_plot", traj_plot)

        # plot_steering_traj(frame_center, trajectory, color=(0,255,0))
        plot_steering_traj(
            frame_center,
            trajectory_rel_2_trajectory(trajectory_rel),
            color=(0, 255, 0),
        )

        frame_merged[
            0 : frame_center.shape[0],
            frame_center.shape[1] : 2 * frame_center.shape[1],
            :,
        ] = frame_center

        # cv2.imshow('frame', cv2.resize(frame_merged, (0,0), fx=0.75, fy=0.75))
        cv2.imshow("frame", frame_center)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
