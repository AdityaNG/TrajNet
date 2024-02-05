import copy
import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from tqdm import tqdm

from ..autopilot_iterator import autopilot_carstate_iterator
from .helper import color_by_index


def get_item_between_timestamp(
    csv_dat, start_ts, end_ts, time_column="time", fault_delay=0.025
):
    """
    Return frame between two given timestamps
    Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
    Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
    """
    ts_dat = csv_dat[csv_dat[time_column].between(start_ts, end_ts)]
    minimum_ts = min(ts_dat[time_column])
    if abs(minimum_ts - start_ts) > fault_delay:
        raise Exception(
            "start_ts is out of bounds: abs(minimum_ts - start_ts) > fault_delay"
        )
    maximum_ts = max(ts_dat[time_column])
    if abs(maximum_ts - end_ts) > fault_delay:
        raise Exception(
            "end_ts is out of bounds: abs(maximum_ts - end_ts) > fault_delay"
        )
    return ts_dat


def get_item_by_timestamp(
    csv_dat, timestamp, time_column="time", fault_delay=0.025
):
    """
    Return frame closest to given timestamp
    Raise exception if delta between timestamp and frame is greaterthan fault_delay (seconds)
    """
    # closest_frames = get_item_between_timestamp(csv_dat, timestamp-fault_delay, timestamp+fault_delay, time_column=time_column, fault_delay=fault_delay)
    closest_frames = get_item_between_timestamp(
        csv_dat,
        timestamp - fault_delay,
        timestamp + fault_delay,
        time_column=time_column,
        fault_delay=float("inf"),
    )
    closest_frames = closest_frames.reset_index(drop=True)
    closest_frame = closest_frames.iloc[
        (closest_frames[time_column] - timestamp).abs().argsort()[0]
    ]
    closest_ts = closest_frame[time_column]
    if abs(timestamp - closest_ts) > fault_delay:
        raise Exception(
            "No such timestamp, fault delay exceeded: abs(timestamp - closest_ts)="
            + str(abs(timestamp - closest_ts))
        )
    return closest_frame


def get_closest_timestamp(timestamp_arr, timestamp, fault_delay=0.065):
    closest_idx = (np.abs(timestamp_arr - timestamp)).argmin()
    closest_ts = timestamp_arr[closest_idx]
    if abs(timestamp - closest_ts) > fault_delay:
        raise Exception(
            "No such timestamp, fault delay exceeded: abs(timestamp - closest_ts)="
            + str(abs(timestamp - closest_ts))
        )
    return closest_ts


def extract_timestamp(file_path, ext=".png"):
    return file_path.split(ext)[0].split("/")[-1]


def compute_freq(timestamp_arr):
    N = timestamp_arr.shape[0]
    tmp1 = timestamp_arr[0 : N - 1]
    tmp2 = timestamp_arr[1:N]
    time_step = tmp2 - tmp1
    mean_time_step = np.mean(time_step)
    mean_freq = 1.0 / mean_time_step
    return mean_freq


def parseIntrinsics(intrinsics_file_path):
    intrinsics_file = open(intrinsics_file_path, "r")
    intrinsics_list = intrinsics_file.read().split("\n\n")
    intrinsics = {}

    sensor_list = [
        '"flir_lwir_camera"',
        '"telops_mwir_camera"',
        '"zed2i_left_camera_optical_frame"',
        '"zed2i_right_camera_optical_frame"',
        '"zed2i_left_camera_optical_frame" (Unrectified / Raw)',
        '"zed2i_right_camera_optical_frame" (Unrectified / Raw)',
    ]

    prams_list = [
        "height",
        "width",
        "distortion_model",
        "D",
        "K",
        "R",
        "P",
    ]

    for sensor_data in intrinsics_list:
        sensor_data_lines = sensor_data.split("\n")
        sensor_id = sensor_data_lines[0]
        yaml_data = "\n".join(sensor_data_lines[1:])
        intrinsics[sensor_id] = yaml.safe_load(yaml_data)

    for sensor_id in sensor_list:
        assert sensor_id in intrinsics, (
            "sensor_id missing from intrinsics " + sensor_id
        )
        for param in prams_list:
            assert param in intrinsics[sensor_id], (
                "params missing from intrinsics[sensor_id] "
                + param
                + " "
                + sensor_id
            )

        intrinsics[sensor_id]["height"] = int(intrinsics[sensor_id]["height"])
        intrinsics[sensor_id]["width"] = int(intrinsics[sensor_id]["width"])

        assert type(intrinsics[sensor_id]["height"]) == int
        assert type(intrinsics[sensor_id]["width"]) == int
        assert type(intrinsics[sensor_id]["distortion_model"]) == str
        assert type(intrinsics[sensor_id]["D"]) == list
        assert type(intrinsics[sensor_id]["K"]) == list
        assert type(intrinsics[sensor_id]["R"]) == list
        assert type(intrinsics[sensor_id]["P"]) == list

        intrinsics[sensor_id]["D"] = np.array(
            intrinsics[sensor_id]["D"]
        ).reshape((5,))
        intrinsics[sensor_id]["K"] = np.array(
            intrinsics[sensor_id]["K"]
        ).reshape((3, 3))
        intrinsics[sensor_id]["R"] = np.array(
            intrinsics[sensor_id]["R"]
        ).reshape((3, 3))
        intrinsics[sensor_id]["P"] = np.array(
            intrinsics[sensor_id]["P"]
        ).reshape((3, 4))

    return intrinsics


class ThermalVoyagerDataset(Dataset):
    def __init__(
        self,
        dataset_path="/home/shared/Thermal_Voyager/Processed/2023-03-20/1/",
        # fault_delay=0.100, # seconds
        fault_delay=0.025,  # seconds
        debug=False,
        recompute_frame_mapping=False,
        **kwargs
    ) -> None:
        self.dataset_path = dataset_path
        self.debug = debug
        self.fault_delay = fault_delay

        self.cache_path = os.path.join(
            os.path.expanduser("~/Thermal_Voyager_cache/"),
            dataset_path.split("Thermal_Voyager/")[-1],
        )
        os.makedirs(self.cache_path, exist_ok=True)
        if self.debug:
            print("cache_path", self.cache_path)

        # self.frame_mapping_csv = os.path.join(self.cache_path, 'FrameMapping.csv')
        self.frame_mapping_csv = os.path.join(
            self.dataset_path, "Mappings.csv"
        )  # No permission to write to dataset

        self.exstrinsics_csv = os.path.join(
            self.dataset_path, "Extrinsics.csv"
        )
        self.objects_detected_csv = os.path.join(
            self.dataset_path, "ObjectsDetected.csv"
        )
        self.poses_csv = os.path.join(self.dataset_path, "Poses.csv")
        self.transforms_csv = os.path.join(self.dataset_path, "Transforms.csv")
        self.intrinsics_txt = os.path.join(self.dataset_path, "Intrinsics.txt")

        self.flir_lwir_raw_folder = os.path.join(
            self.dataset_path, "flir_lwir", "image_raw"
        )
        self.telops_mwir_raw_folder = os.path.join(
            self.dataset_path, "telops_mwir", "image_raw"
        )
        self.flir_lwir_folder = os.path.join(
            self.dataset_path, "flir_lwir", "image_raw_histogram_equalized"
        )
        self.telops_mwir_folder = os.path.join(
            self.dataset_path, "telops_mwir", "image_raw_histogram_equalized"
        )
        self.velodyne_points_folder = os.path.join(
            self.dataset_path, "velodyne_points"
        )

        self.zed2i_folder = os.path.join(
            self.dataset_path, "zed2i", "zed_node"
        )
        self.zed2i_point_cloud_folder = os.path.join(
            self.zed2i_folder, "point_cloud", "cloud_registered"
        )
        # self.zed2i_stereo_folder = os.path.join(self.zed2i_folder, 'stereo', 'image_rect_color')
        self.zed2i_stereo_folder = os.path.join(
            self.zed2i_folder, "stereo_raw", "image_raw_color"
        )
        self.zed2i_stereo_raw_folder = os.path.join(
            self.zed2i_folder, "stereo_raw", "image_raw_color"
        )

        #######################
        # Check if files and folders exist
        assert os.path.isfile(self.exstrinsics_csv), (
            "exstrinsics_csv missing " + self.exstrinsics_csv
        )
        assert os.path.isfile(self.objects_detected_csv), (
            "objects_detected_csv missing " + self.objects_detected_csv
        )
        assert os.path.isfile(self.poses_csv), (
            "poses_csv missing " + self.poses_csv
        )
        assert os.path.isfile(self.transforms_csv), (
            "transforms_csv missing " + self.transforms_csv
        )
        assert os.path.isfile(self.intrinsics_txt), (
            "intrinsics_txt missing " + self.intrinsics_txt
        )

        assert os.path.isdir(self.flir_lwir_folder), (
            "flir_lwir_folder missing " + self.flir_lwir_folder
        )
        assert os.path.isdir(self.telops_mwir_folder), (
            "telops_mwir_folder missing " + self.telops_mwir_folder
        )
        assert os.path.isdir(self.flir_lwir_raw_folder), (
            "flir_lwir_raw_folder missing " + self.flir_lwir_raw_folder
        )
        assert os.path.isdir(self.telops_mwir_raw_folder), (
            "telops_mwir_raw_folder missing " + self.telops_mwir_raw_folder
        )
        assert os.path.isdir(self.velodyne_points_folder), (
            "velodyne_points_folder missing " + self.velodyne_points_folder
        )

        assert os.path.isdir(self.zed2i_folder), (
            "zed2i_folder missing " + self.zed2i_folder
        )
        assert os.path.isdir(self.zed2i_point_cloud_folder), (
            "zed2i_point_cloud_folder missing " + self.zed2i_point_cloud_folder
        )
        assert os.path.isdir(self.zed2i_stereo_folder), (
            "zed2i_stereo_folder missing " + self.zed2i_stereo_folder
        )
        assert os.path.isdir(self.zed2i_stereo_raw_folder), (
            "zed2i_stereo_raw_folder missing " + self.zed2i_stereo_raw_folder
        )
        #######################
        # Load in the CSVs
        self.exstrinsics_df = pd.read_csv(self.exstrinsics_csv)
        # self.objects_detected_df = pd.read_csv(self.objects_detected_csv)
        self.poses_df = pd.read_csv(self.poses_csv)
        self.transforms_df = pd.read_csv(self.transforms_csv)
        self.intrinsics_dict = parseIntrinsics(self.intrinsics_txt)

        assert len(self.exstrinsics_df.index) > 0, (
            "exstrinsics_df is empty " + self.exstrinsics_csv
        )
        # assert len(self.objects_detected_df.index) > 0, "objects_detected_df is empty " + self.objects_detected_csv
        assert len(self.poses_df.index) > 0, (
            "poses_df is empty " + self.poses_csv
        )
        assert len(self.transforms_df.index) > 0, (
            "transforms_df is empty " + self.transforms_csv
        )

        # self.objects_detected_df['time'] = pd.to_numeric(self.objects_detected_df['time'])
        # self.poses_df['time'] = pd.to_numeric(self.poses_df['time'])
        # self.transforms_df['time'] = pd.to_numeric(self.transforms_df['time'])

        self.zedCenter2velo = self.exstrinsics_df.loc[
            (self.exstrinsics_df["parent_frame"] == "zed2i_camera_center")
            & (self.exstrinsics_df["child_frame"] == "velodyne")
        ]
        self.zedCenter2zedLeftFrame = self.exstrinsics_df.loc[
            (self.exstrinsics_df["parent_frame"] == "zed2i_camera_center")
            & (self.exstrinsics_df["child_frame"] == "zed2i_left_camera_frame")
        ]
        self.zedLeftFrame2zedLeftOpticalFrame = self.exstrinsics_df.loc[
            (self.exstrinsics_df["parent_frame"] == "zed2i_left_camera_frame")
            & (
                self.exstrinsics_df["child_frame"]
                == "zed2i_left_camera_optical_frame"
            )
        ]
        self.netTransformZedLeftOptical2velodyne = (
            -np.array(
                (
                    self.zedLeftFrame2zedLeftOpticalFrame.translation_x,
                    self.zedLeftFrame2zedLeftOpticalFrame.translation_y,
                    self.zedLeftFrame2zedLeftOpticalFrame.translation_z,
                )
            )
            - np.array(
                (
                    self.zedCenter2zedLeftFrame.translation_x,
                    self.zedCenter2zedLeftFrame.translation_y,
                    self.zedCenter2zedLeftFrame.translation_z,
                )
            )
            + np.array(
                (
                    self.zedCenter2velo.translation_x,
                    self.zedCenter2velo.translation_y,
                    self.zedCenter2velo.translation_z,
                )
            )
        )
        #######################
        # Get file list
        self.flir_lwir_folder_list = glob.glob(
            os.path.join(self.flir_lwir_folder, "*.png")
        )
        self.telops_mwir_folder_list = glob.glob(
            os.path.join(self.telops_mwir_folder, "*.png")
        )
        self.flir_lwir_raw_folder_list = glob.glob(
            os.path.join(self.flir_lwir_raw_folder, "*.png")
        )
        self.telops_mwir_raw_folder_list = glob.glob(
            os.path.join(self.telops_mwir_raw_folder, "*.png")
        )
        self.velodyne_points_folder_list = glob.glob(
            os.path.join(self.velodyne_points_folder, "*.pcd")
        )

        self.zed2i_point_cloud_folder_list = glob.glob(
            os.path.join(self.zed2i_point_cloud_folder, "*.pcd")
        )
        self.zed2i_stereo_folder_list = glob.glob(
            os.path.join(self.zed2i_stereo_folder, "*.png")
        )
        self.zed2i_stereo_raw_folder_list = glob.glob(
            os.path.join(self.zed2i_stereo_raw_folder, "*.png")
        )

        assert len(self.flir_lwir_folder_list) > 0, (
            "flir_lwir_folder_list is empty " + self.flir_lwir_folder
        )
        assert len(self.telops_mwir_folder_list) > 0, (
            "telops_mwir_folder_list is empty " + self.telops_mwir_folder
        )
        assert len(self.flir_lwir_raw_folder_list) > 0, (
            "flir_lwir_raw_folder_list is empty "
            + self.flir_lwir_raw_folder_list
        )
        assert len(self.telops_mwir_raw_folder_list) > 0, (
            "telops_mwir_raw_folder_list is empty "
            + self.telops_mwir_raw_folder_list
        )
        assert len(self.velodyne_points_folder_list) > 0, (
            "velodyne_points_folder_list is empty "
            + self.velodyne_points_folder
        )
        assert len(self.zed2i_point_cloud_folder_list) > 0, (
            "zed2i_point_cloud_folder_list is empty "
            + self.zed2i_point_cloud_folder
        )
        assert len(self.zed2i_stereo_folder_list) > 0, (
            "zed2i_stereo_folder_list is empty " + self.zed2i_stereo_folder
        )
        assert len(self.zed2i_stereo_raw_folder_list) > 0, (
            "zed2i_stereo_raw_folder_list is empty "
            + self.zed2i_stereo_raw_folder
        )
        #######################
        # Load all timestamps stored in nanoseconds
        # self.objects_detected_timestamps = np.array(self.objects_detected_df['time'].values.tolist(), dtype=int)
        self.poses_timestamps = np.array(
            self.poses_df["time"].values.tolist(), dtype=int
        )
        self.transforms_timestamps = np.array(
            self.transforms_df["time"].values.tolist(), dtype=int
        )
        self.flir_lwir_timestamps = np.array(
            sorted(list(map(extract_timestamp, self.flir_lwir_folder_list))),
            dtype=int,
        )
        self.telops_mwir_timestamps = np.array(
            sorted(list(map(extract_timestamp, self.telops_mwir_folder_list))),
            dtype=int,
        )
        self.flir_lwir_raw_timestamps = np.array(
            sorted(
                list(map(extract_timestamp, self.flir_lwir_raw_folder_list))
            ),
            dtype=int,
        )
        self.telops_mwir_raw_timestamps = np.array(
            sorted(
                list(map(extract_timestamp, self.telops_mwir_raw_folder_list))
            ),
            dtype=int,
        )
        self.velodyne_points_timestamps = np.array(
            sorted(
                list(
                    map(
                        lambda p: extract_timestamp(p, ext=".pcd"),
                        self.velodyne_points_folder_list,
                    )
                )
            ),
            dtype=int,
        )
        self.zed2i_point_cloud_timestamps = np.array(
            sorted(
                list(
                    map(
                        lambda p: extract_timestamp(p, ext=".pcd"),
                        self.zed2i_point_cloud_folder_list,
                    )
                )
            ),
            dtype=int,
        )
        self.zed2i_stereo_timestamps = np.array(
            sorted(
                list(map(extract_timestamp, self.zed2i_stereo_folder_list))
            ),
            dtype=int,
        )
        self.zed2i_stereo_raw_timestamps = np.array(
            sorted(
                list(map(extract_timestamp, self.zed2i_stereo_raw_folder_list))
            ),
            dtype=int,
        )

        # self.objects_detected_freq = compute_freq(self.objects_detected_timestamps / 10**9)
        self.poses_freq = compute_freq(self.poses_timestamps / 10**9)
        self.transforms_freq = compute_freq(
            self.transforms_timestamps / 10**9
        )
        self.flir_lwir_freq = compute_freq(self.flir_lwir_timestamps / 10**9)
        self.telops_mwir_freq = compute_freq(
            self.telops_mwir_timestamps / 10**9
        )
        self.flir_lwir_raw_freq = compute_freq(
            self.flir_lwir_raw_timestamps / 10**9
        )
        self.telops_mwir_raw_freq = compute_freq(
            self.telops_mwir_raw_timestamps / 10**9
        )
        self.velodyne_points_freq = compute_freq(
            self.velodyne_points_timestamps / 10**9
        )
        self.zed2i_point_cloud_freq = compute_freq(
            self.zed2i_point_cloud_timestamps / 10**9
        )
        self.zed2i_stereo_freq = compute_freq(
            self.zed2i_stereo_timestamps / 10**9
        )
        self.zed2i_stereo_raw_freq = compute_freq(
            self.zed2i_stereo_raw_timestamps / 10**9
        )

        if self.debug:
            # print('objects_detected_freq',  self.objects_detected_freq)
            print("poses_freq", self.poses_freq)
            print("transforms_freq", self.transforms_freq)
            print("flir_lwir_freq", self.flir_lwir_freq)
            print("telops_mwir_freq", self.telops_mwir_freq)
            print("flir_lwir_raw_freq", self.flir_lwir_raw_freq)
            print("telops_mwir_raw_freq", self.telops_mwir_raw_freq)
            print("velodyne_points_freq", self.velodyne_points_freq)
            print("zed2i_point_cloud_freq", self.zed2i_point_cloud_freq)
            print("zed2i_stereo_freq", self.zed2i_stereo_freq)
            print("zed2i_stereo_raw_freq", self.zed2i_stereo_raw_freq)

        # Compute intersection
        self.timestamp_lb = max(
            # min(self.objects_detected_timestamps),
            min(self.poses_timestamps),
            min(self.transforms_timestamps),
            min(self.flir_lwir_timestamps),
            min(self.telops_mwir_timestamps),
            min(self.flir_lwir_raw_timestamps),
            min(self.telops_mwir_raw_timestamps),
            min(self.velodyne_points_timestamps),
            min(self.zed2i_point_cloud_timestamps),
            min(self.zed2i_stereo_timestamps),
            min(self.zed2i_stereo_raw_timestamps),
        )

        self.timestamp_ub = min(
            # max(self.objects_detected_timestamps),
            max(self.poses_timestamps),
            max(self.transforms_timestamps),
            max(self.flir_lwir_timestamps),
            max(self.telops_mwir_timestamps),
            max(self.flir_lwir_raw_timestamps),
            max(self.telops_mwir_raw_timestamps),
            max(self.velodyne_points_timestamps),
            max(self.zed2i_point_cloud_timestamps),
            max(self.zed2i_stereo_timestamps),
            max(self.zed2i_stereo_raw_timestamps),
        )

        # LiDAR is the master sensor
        self.dataset_freq = self.velodyne_points_freq
        self.dataset_timestamps = self.velodyne_points_timestamps[
            np.logical_and(
                self.timestamp_lb < self.velodyne_points_timestamps,
                self.velodyne_points_timestamps < self.timestamp_ub,
            )
        ]

        if (
            os.path.isfile(self.frame_mapping_csv)
            and not recompute_frame_mapping
        ):
            self.frame_mapping_df = pd.read_csv(self.frame_mapping_csv)

            if "/velodyne_points" in self.frame_mapping_df:
                self.frame_mapping_df[
                    "dataset_timestamps"
                ] = self.frame_mapping_df["/velodyne_points"]
                self.dataset_timestamps = self.frame_mapping_df[
                    "dataset_timestamps"
                ].copy()

            if "/flir_lwir/image_raw" in self.frame_mapping_df:
                self.frame_mapping_df[
                    "flir_lwir_timestamps"
                ] = self.frame_mapping_df["/flir_lwir/image_raw"]
            if "/telops_mwir/image_raw" in self.frame_mapping_df:
                self.frame_mapping_df[
                    "telops_mwir_timestamps"
                ] = self.frame_mapping_df["/telops_mwir/image_raw"]
            if "/flir_lwir/image_raw" in self.frame_mapping_df:
                self.frame_mapping_df[
                    "flir_lwir_raw_timestamps"
                ] = self.frame_mapping_df["/flir_lwir/image_raw"]
            if "/telops_mwir/image_raw" in self.frame_mapping_df:
                self.frame_mapping_df[
                    "telops_mwir_raw_timestamps"
                ] = self.frame_mapping_df["/telops_mwir/image_raw"]
            if "/velodyne_points" in self.frame_mapping_df:
                self.frame_mapping_df[
                    "velodyne_points_timestamps"
                ] = self.frame_mapping_df["/velodyne_points"]
            if (
                "/zed2i/zed_node/point_cloud/cloud_registered"
                in self.frame_mapping_df
            ):
                self.frame_mapping_df[
                    "zed2i_point_cloud_timestamps"
                ] = self.frame_mapping_df[
                    "/zed2i/zed_node/point_cloud/cloud_registered"
                ]
            if (
                "/zed2i/zed_node/stereo_raw/image_raw_color"
                in self.frame_mapping_df
            ):
                self.frame_mapping_df[
                    "zed2i_stereo_timestamps"
                ] = self.frame_mapping_df[
                    "/zed2i/zed_node/stereo_raw/image_raw_color"
                ]
            if (
                "/zed2i/zed_node/stereo_raw/image_raw_color"
                in self.frame_mapping_df
            ):
                self.frame_mapping_df[
                    "zed2i_stereo_raw_timestamps"
                ] = self.frame_mapping_df[
                    "/zed2i/zed_node/stereo_raw/image_raw_color"
                ]
            # if '' in self.frame_mapping_df:
            #     self.frame_mapping_df['poses_df'] = self.frame_mapping_df['']
            # if '' in self.frame_mapping_df:
            #     self.frame_mapping_df['transforms_df'] = self.frame_mapping_df['']

            # self.dataset_timestamps = np.array(self.frame_mapping_df['dataset_timestamps'].tolist())
            valid_timestamps = np.ones_like(
                self.dataset_timestamps, dtype=bool
            )
            for col in [
                "flir_lwir_timestamps",
                "telops_mwir_timestamps",
                "flir_lwir_raw_timestamps",
                "telops_mwir_raw_timestamps",
                "velodyne_points_timestamps",
                "zed2i_point_cloud_timestamps",
                "zed2i_stereo_timestamps",
                "zed2i_stereo_raw_timestamps",
            ]:
                # print('col', col)
                time_deltas = abs(
                    self.dataset_timestamps - self.frame_mapping_df[col]
                )
                # print('min, max', min(time_deltas), max(time_deltas))
                valid_timestamps = np.logical_and(
                    valid_timestamps, time_deltas < self.fault_delay * 10**9
                )

            self.frame_mapping_df["valid_timestamps"] = valid_timestamps
        else:
            self.frame_mapping_df = {
                "dataset_timestamps": [],
                "flir_lwir_timestamps": [],
                "telops_mwir_timestamps": [],
                "flir_lwir_raw_timestamps": [],
                "telops_mwir_raw_timestamps": [],
                "velodyne_points_timestamps": [],
                "zed2i_point_cloud_timestamps": [],
                "zed2i_stereo_timestamps": [],
                "zed2i_stereo_raw_timestamps": [],
                "poses_df": [],
                "transforms_df": [],
                "valid_timestamps": [],
                # self.objects_detected_df, # Omit objects_detected_df as not all frames have objects
            }
            self.frame_mapping_df["valid_timestamps"] = np.ones_like(
                self.dataset_timestamps, dtype=bool
            )

            self.sensor_timestamp_list = {
                "flir_lwir_timestamps": self.flir_lwir_timestamps,
                "telops_mwir_timestamps": self.telops_mwir_timestamps,
                "flir_lwir_raw_timestamps": self.flir_lwir_raw_timestamps,
                "telops_mwir_raw_timestamps": self.telops_mwir_raw_timestamps,
                "velodyne_points_timestamps": self.velodyne_points_timestamps,
                "zed2i_point_cloud_timestamps": self.zed2i_point_cloud_timestamps,
                "zed2i_stereo_timestamps": self.zed2i_stereo_timestamps,
                "zed2i_stereo_raw_timestamps": self.zed2i_stereo_raw_timestamps,
            }

            self.dataframe_list = {
                # self.objects_detected_df, # Omit objects_detected_df as not all frames have objects
                # 'poses_df': self.poses_df,
                # 'transforms_df': self.transforms_df,
            }

            for timestamp_index in tqdm(range(len(self.dataset_timestamps))):
                timestamp_ns = self.dataset_timestamps[timestamp_index]
                self.frame_mapping_df["dataset_timestamps"] += [timestamp_ns]
                for sensor_id in self.sensor_timestamp_list:
                    sensor = self.sensor_timestamp_list[sensor_id]
                    ts = get_closest_timestamp(
                        sensor, timestamp_ns, fault_delay=float("inf")
                    )
                    self.frame_mapping_df[sensor_id] += [ts]

                    if abs(timestamp_ns - ts) > self.fault_delay * 10**9:
                        self.frame_mapping_df["valid_timestamps"][
                            timestamp_index
                        ] = False

                for df_id in self.dataframe_list:
                    df = self.dataframe_list[df_id]
                    ts = get_item_by_timestamp(
                        df, timestamp_ns, fault_delay=float("inf")
                    )["time"]
                    self.frame_mapping_df[df_id] += [ts]

                    if abs(timestamp_ns - ts) > self.fault_delay * 10**9:
                        self.frame_mapping_df["valid_timestamps"][
                            timestamp_index
                        ] = False

            if self.debug:
                for data_id in self.frame_mapping_df:
                    print(data_id, "->", len(self.frame_mapping_df[data_id]))

            self.frame_mapping_df = pd.DataFrame(self.frame_mapping_df)
            self.frame_mapping_df.to_csv(self.frame_mapping_csv)

        self.frames_dropped = len(
            self.frame_mapping_df["valid_timestamps"]
        ) - np.sum(self.frame_mapping_df["valid_timestamps"])
        if self.debug:
            print("self.frames_dropped", self.frames_dropped)
        self.dataset_timestamps = self.dataset_timestamps[
            self.frame_mapping_df["valid_timestamps"]
        ]

        self.dataset_timestamps = self.dataset_timestamps.tolist()
        self.size = len(self.dataset_timestamps)

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
        timestamp_ns = self.dataset_timestamps[key]
        timestamp = timestamp_ns / 10**9

        # Not all frames have objects
        # objects_detected_data = get_item_by_timestamp(self.objects_detected_df, timestamp_ns, fault_delay=self.fault_delay * 10**9)
        # poses_data = get_item_by_timestamp(self.poses_df, timestamp_ns, fault_delay=self.fault_delay * 10**9)
        # transforms_data = get_item_by_timestamp(self.transforms_df, timestamp_ns, fault_delay=self.fault_delay * 10**9)

        flir_lwir_timestamp = get_closest_timestamp(
            self.flir_lwir_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        telops_mwir_timestamp = get_closest_timestamp(
            self.telops_mwir_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        flir_lwir_raw_timestamp = get_closest_timestamp(
            self.flir_lwir_raw_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        telops_mwir_raw_timestamp = get_closest_timestamp(
            self.telops_mwir_raw_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        velodyne_points_timestamp = get_closest_timestamp(
            self.velodyne_points_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        zed2i_point_cloud_timestamp = get_closest_timestamp(
            self.zed2i_point_cloud_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        zed2i_stereo_timestamp = get_closest_timestamp(
            self.zed2i_stereo_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )
        zed2i_stereo_raw_timestamp = get_closest_timestamp(
            self.zed2i_stereo_raw_timestamps,
            timestamp_ns,
            fault_delay=self.fault_delay * 10**9,
        )

        flir_lwir_data_path = os.path.join(
            self.flir_lwir_folder, str(flir_lwir_timestamp) + ".png"
        )
        telops_mwir_data_path = os.path.join(
            self.telops_mwir_folder, str(telops_mwir_timestamp) + ".png"
        )
        flir_lwir_raw_data_path = os.path.join(
            self.flir_lwir_raw_folder, str(flir_lwir_raw_timestamp) + ".png"
        )
        telops_mwir_raw_data_path = os.path.join(
            self.telops_mwir_raw_folder,
            str(telops_mwir_raw_timestamp) + ".png",
        )
        velodyne_points_data_path = os.path.join(
            self.velodyne_points_folder,
            str(velodyne_points_timestamp) + ".pcd",
        )
        zed2i_point_cloud_data_path = os.path.join(
            self.zed2i_point_cloud_folder,
            str(zed2i_point_cloud_timestamp) + ".pcd",
        )
        zed2i_stereo_data_path = os.path.join(
            self.zed2i_stereo_folder, str(zed2i_stereo_timestamp) + ".png"
        )
        zed2i_stereo_raw_data_path = os.path.join(
            self.zed2i_stereo_raw_folder,
            str(zed2i_stereo_raw_timestamp) + ".png",
        )

        assert os.path.isfile(flir_lwir_data_path), (
            "flir_lwir_data_path missing " + flir_lwir_data_path
        )
        assert os.path.isfile(telops_mwir_data_path), (
            "telops_mwir_data_path missing " + telops_mwir_data_path
        )
        assert os.path.isfile(flir_lwir_raw_data_path), (
            "flir_lwir_raw_data_path missing " + flir_lwir_raw_data_path
        )
        assert os.path.isfile(telops_mwir_raw_data_path), (
            "telops_mwir_raw_data_path missing " + telops_mwir_raw_data_path
        )
        assert os.path.isfile(velodyne_points_data_path), (
            "velodyne_points_data_path missing " + velodyne_points_data_path
        )
        assert os.path.isfile(zed2i_point_cloud_data_path), (
            "zed2i_point_cloud_data_path missing "
            + zed2i_point_cloud_data_path
        )
        assert os.path.isfile(zed2i_stereo_data_path), (
            "zed2i_stereo_data_path missing " + zed2i_stereo_data_path
        )
        assert os.path.isfile(zed2i_stereo_raw_data_path), (
            "zed2i_stereo_raw_data_path missing " + zed2i_stereo_raw_data_path
        )

        return {
            "timestamp": timestamp,
            # 'objects_detected_data': objects_detected_data,
            # 'poses_data': poses_data,
            # 'transforms_data': transforms_data,
            "flir_lwir_data_path": flir_lwir_data_path,
            "telops_mwir_data_path": telops_mwir_data_path,
            "flir_lwir_raw_data_path": flir_lwir_raw_data_path,
            "telops_mwir_raw_data_path": telops_mwir_raw_data_path,
            "velodyne_points_data_path": velodyne_points_data_path,
            "zed2i_point_cloud_data_path": zed2i_point_cloud_data_path,
            "zed2i_stereo_data_path": zed2i_stereo_data_path,
            "zed2i_stereo_raw_data_path": zed2i_stereo_raw_data_path,
        }


ZED_2_BASE = [
    "zed2i_left_camera_optical_frame",
    "zed2i_left_camera_frame",
    "zed2i_camera_center",
    "zed2i_base_link",
    "base_link",
]

LIDAR_2_BASE = [
    "velodyne",
    "zed2i_camera_center",
    "zed2i_base_link",
    "base_link",
]

FLIR_2_BASE = [
    "flir_lwir_camera",
    "zed2i_camera_center",
    "zed2i_base_link",
    "base_link",
]

TELOPS_2_BASE = [
    "telops_mwir_camera",
    "zed2i_camera_center",
    "zed2i_base_link",
    "base_link",
]


class ThermalVoyager_ThermalDepthDataset(ThermalVoyagerDataset):
    def __init__(self, *args, **kwargs):
        super(ThermalVoyager_ThermalDepthDataset, self).__init__(
            *args, **kwargs
        )
        self.img_transform = kwargs["transform"]

        self.lidar2base = self.get_transformation(LIDAR_2_BASE)
        self.zed2base = self.get_transformation(ZED_2_BASE)
        self.flir2base = self.get_transformation(FLIR_2_BASE)
        self.base2flir = np.linalg.inv(self.flir2base)
        self.telops2base = self.get_transformation(TELOPS_2_BASE)
        self.base2telops = np.linalg.inv(self.telops2base)
        self.zed2flir = self.zed2base @ self.base2flir
        self.lidar2flir = self.lidar2base @ self.base2flir
        self.zed2telops = self.zed2base @ self.base2telops
        self.lidar2telops = self.lidar2base @ self.base2telops

    def __len__(
        self,
    ):
        original_len = super().__len__()
        # return original_len*2
        return original_len

    def get_transformation(self, frame_jumps):
        transformation = np.eye(4, 4)

        for index in range(1, len(frame_jumps)):
            child_frame = frame_jumps[index - 1]
            parent_frame = frame_jumps[index]
            transforms_df = self.exstrinsics_df.loc[
                (self.exstrinsics_df["parent_frame"] == parent_frame)
                & (self.exstrinsics_df["child_frame"] == child_frame)
            ]

            # print('child_frame', child_frame)
            # print('parent_frame', parent_frame)
            # print('type(transforms_df)', type(transforms_df))

            transformation_new = np.eye(4, 4)

            T = np.array(
                [
                    transforms_df.translation_x,
                    transforms_df.translation_y,
                    transforms_df.translation_z,
                ]
            ).reshape((3,))
            R = Rotation.from_quat(
                np.array(
                    [
                        transforms_df.rotation_x,
                        transforms_df.rotation_y,
                        transforms_df.rotation_z,
                        transforms_df.rotation_w,
                    ]
                ).reshape(
                    4,
                )
            ).as_matrix()

            transformation_new[:3, :3] = R
            transformation_new[:3, 3] = T

            transformation = transformation @ transformation_new

        return transformation

    def draw_points(self, u_p, v_p, img, color, size=4):
        if u_p.shape[0] == 0:
            return img
        color = np.array(color)
        if color.shape[0] != u_p.shape[0]:
            assert color.shape[0] == 3, "Color is not in RGB format"
            color = np.tile(color, (u_p.shape[0], 1))
        color = color.astype(np.uint8)

        for i in range(u_p.shape[0]):
            u, v = u_p[i], v_p[i]
            cv2.circle(img, (int(u), int(v)), size, color[i, :].tolist(), -1)

        return img

    def preprocess_points(self, lidarPCD, zedPCD):
        if not lidarPCD.has_colors():
            z = np.asarray(lidarPCD.points)[:, 0]
            z_normalized = (z - z.min()) / (z.max() - z.min())
            colors = plt.cm.jet(z_normalized)[:, :3]
            lidarPCD.colors = o3d.utility.Vector3dVector(colors[:, :3])
        else:
            print("LIDAR point cloud has color!!")
            colors = np.asarray(lidarPCD.colors)

        # Filter the points based on their azimuth angle
        threshold_angle = 51
        points = np.asarray(lidarPCD.points)
        azimuth_angle = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
        forward_points = points[
            (azimuth_angle >= -threshold_angle)
            & (azimuth_angle <= threshold_angle)
        ]
        forward_colors = colors[
            (azimuth_angle >= -threshold_angle)
            & (azimuth_angle <= threshold_angle)
        ]
        lidarPCD.points = o3d.utility.Vector3dVector(forward_points)
        lidarPCD.colors = o3d.utility.Vector3dVector(forward_colors)

        lidarPCD.rotate(
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), center=(0, 0, 0)
        )
        zedPCD.rotate(
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), center=(0, 0, 0)
        )

        lidarPoints = np.asarray(lidarPCD.points)
        zedPoints = np.asarray(zedPCD.points)

        lidarPoints[:, [1, 2]] = lidarPoints[:, [2, 1]]
        lidarPoints[:, 1] *= -1

        zedPoints[:, [1, 2]] = zedPoints[:, [2, 1]]
        zedPoints[:, 1] *= -1

        lidarPCD.points = o3d.utility.Vector3dVector(lidarPoints)
        zedPCD.points = o3d.utility.Vector3dVector(zedPoints)

    def __getitem__(self, frame_index_original):
        original_len = super().__len__()
        frame_index = frame_index_original % original_len
        flip_flag = bool(frame_index_original // original_len)

        frame = super().__getitem__(frame_index)

        zed2i_stereo_np = cv2.imread(frame["zed2i_stereo_data_path"])
        zed2i_left_np = zed2i_stereo_np[:, : zed2i_stereo_np.shape[1] // 2, :]
        zed2i_right_np = zed2i_stereo_np[:, zed2i_stereo_np.shape[1] // 2 :, :]
        if self.debug:
            print("zed2i_stereo_np.shape", zed2i_stereo_np.shape)

        thermal_lwir_np = cv2.imread(frame["flir_lwir_data_path"])
        thermal_mwir_np = cv2.imread(frame["telops_mwir_data_path"])

        thermal_lwir_raw_np = cv2.imread(frame["flir_lwir_raw_data_path"])
        thermal_mwir_raw_np = cv2.imread(frame["telops_mwir_raw_data_path"])

        thermal_lwir_color_np = thermal_lwir_np.copy()
        thermal_mwir_color_np = thermal_mwir_np.copy()

        thermal_lwir_np_shape = (
            self.intrinsics_dict['"flir_lwir_camera"']["height"],
            self.intrinsics_dict['"flir_lwir_camera"']["width"],
        )
        assert thermal_lwir_np.shape[:2] == thermal_lwir_np_shape, (
            "Shape mismatch, "
            + str(thermal_lwir_np_shape)
            + ", "
            + str(thermal_lwir_np.shape[:2])
        )

        thermal_mwir_np_shape = (
            self.intrinsics_dict['"telops_mwir_camera"']["height"],
            self.intrinsics_dict['"telops_mwir_camera"']["width"],
        )
        assert thermal_mwir_np.shape[:2] == thermal_mwir_np_shape, (
            "Shape mismatch, "
            + str(thermal_mwir_np_shape)
            + ", "
            + str(thermal_mwir_np.shape[:2])
        )
        # thermal_mwir_np = cv2.rotate(thermal_mwir_np, cv2.ROTATE_180)

        # thermal_lwir_np = thermal_lwir_np[:,:,0]
        # thermal_mwir_np = thermal_mwir_np[:,:,0]

        lidarPCD = o3d.io.read_point_cloud(frame["velodyne_points_data_path"])
        zedPCD = o3d.io.read_point_cloud(frame["zed2i_point_cloud_data_path"])

        lidarPCD = lidarPCD.remove_non_finite_points()
        zedPCD = zedPCD.remove_non_finite_points()

        # self.preprocess_points(lidarPCD, zedPCD)

        zed_depth_lwir = np.zeros(
            (
                self.intrinsics_dict['"flir_lwir_camera"']["height"],
                self.intrinsics_dict['"flir_lwir_camera"']["width"],
            )
        ).astype(float)
        zed_depth_mwir = np.zeros(
            (
                self.intrinsics_dict['"telops_mwir_camera"']["height"],
                self.intrinsics_dict['"telops_mwir_camera"']["width"],
            )
        ).astype(float)
        lidar_depth_lwir = np.zeros(
            (
                self.intrinsics_dict['"flir_lwir_camera"']["height"],
                self.intrinsics_dict['"flir_lwir_camera"']["width"],
            )
        ).astype(float)
        lidar_depth_mwir = np.zeros(
            (
                self.intrinsics_dict['"telops_mwir_camera"']["height"],
                self.intrinsics_dict['"telops_mwir_camera"']["width"],
            )
        ).astype(float)

        zed_disparity_lwir = np.zeros(
            (
                self.intrinsics_dict['"flir_lwir_camera"']["height"],
                self.intrinsics_dict['"flir_lwir_camera"']["width"],
            )
        ).astype(float)
        zed_disparity_mwir = np.zeros(
            (
                self.intrinsics_dict['"telops_mwir_camera"']["height"],
                self.intrinsics_dict['"telops_mwir_camera"']["width"],
            )
        ).astype(float)
        lidar_disparity_lwir = np.zeros(
            (
                self.intrinsics_dict['"flir_lwir_camera"']["height"],
                self.intrinsics_dict['"flir_lwir_camera"']["width"],
            )
        ).astype(float)
        lidar_disparity_mwir = np.zeros(
            (
                self.intrinsics_dict['"telops_mwir_camera"']["height"],
                self.intrinsics_dict['"telops_mwir_camera"']["width"],
            )
        ).astype(float)

        zed_depth_lwir_mask = np.zeros(
            (
                self.intrinsics_dict['"flir_lwir_camera"']["height"],
                self.intrinsics_dict['"flir_lwir_camera"']["width"],
            )
        ).astype(bool)
        zed_depth_mwir_mask = np.zeros(
            (
                self.intrinsics_dict['"telops_mwir_camera"']["height"],
                self.intrinsics_dict['"telops_mwir_camera"']["width"],
            )
        ).astype(bool)
        lidar_depth_lwir_mask = np.zeros(
            (
                self.intrinsics_dict['"flir_lwir_camera"']["height"],
                self.intrinsics_dict['"flir_lwir_camera"']["width"],
            )
        ).astype(bool)
        lidar_depth_mwir_mask = np.zeros(
            (
                self.intrinsics_dict['"telops_mwir_camera"']["height"],
                self.intrinsics_dict['"telops_mwir_camera"']["width"],
            )
        ).astype(bool)

        # telops2lidar = np.linalg.inv(self.lidar2telops)
        # telops2lidar = np.linalg.inv(self.lidar2telops)

        # zedFlirPCD = copy.deepcopy(zedPCD).transform(np.linalg.inv(self.zed2flir))
        # lidarFlirPCD = copy.deepcopy(lidarPCD).transform(np.linalg.inv(self.lidar2flir))
        # zedTelopsPCD = copy.deepcopy(zedPCD).transform(np.linalg.inv(self.zed2telops))
        # lidarTelopsPCD = copy.deepcopy(lidarPCD).transform(np.linalg.inv(self.lidar2telops))

        zedFlirPCD = copy.deepcopy(zedPCD).transform(self.zed2flir)
        lidarFlirPCD = copy.deepcopy(lidarPCD).transform(self.lidar2flir)
        zedTelopsPCD = copy.deepcopy(zedPCD).transform(self.zed2telops)
        lidarTelopsPCD = copy.deepcopy(lidarPCD).transform(self.lidar2telops)

        lwir_lidarFlirPoints_np = np.asarray(lidarFlirPCD.points)
        lwir_zedFlirPoints_np = np.asarray(zedFlirPCD.points)
        mwir_lidarTelopsPoints_np = np.asarray(lidarTelopsPCD.points)
        mwir_zedTelopsPoints_np = np.asarray(zedTelopsPCD.points)

        # Project zed points onto lwir
        zedPointsFlir_np, jacobian_flir = cv2.projectPoints(
            lwir_zedFlirPoints_np,
            np.eye(3, 3),
            np.zeros((3, 1)),
            self.intrinsics_dict['"flir_lwir_camera"']["K"],
            self.intrinsics_dict['"flir_lwir_camera"']["D"],
        )
        zedPointsFlir_np = zedPointsFlir_np.reshape((-1, 2)).astype(int)
        zed_lwir_filter = np.logical_and(
            np.logical_and(
                0 <= zedPointsFlir_np[:, 0],
                zedPointsFlir_np[:, 0]
                < self.intrinsics_dict['"flir_lwir_camera"']["width"],
            ),
            np.logical_and(
                0 <= zedPointsFlir_np[:, 1],
                zedPointsFlir_np[:, 1]
                < self.intrinsics_dict['"flir_lwir_camera"']["height"],
            ),
        )
        zedPointsFlir_np = zedPointsFlir_np[zed_lwir_filter]
        lwir_zedFlirPoints_np = lwir_zedFlirPoints_np[zed_lwir_filter]
        u = zedPointsFlir_np[:, 0]
        v = zedPointsFlir_np[:, 1]

        zed_depth_lwir[v, u] = lwir_zedFlirPoints_np[:, 2]
        zed_disparity_lwir[v, u] = np.reciprocal(zed_depth_lwir[v, u])
        zed_depth_lwir_mask[v, u] = True

        # self.draw_points(u, v, thermal_lwir_np, color_by_index(lwir_zedFlirPoints_np,2))
        self.draw_points(
            u,
            v,
            thermal_lwir_color_np,
            color_by_index(
                lwir_zedFlirPoints_np, 2, min_height=0, max_height=50
            ),
        )
        del u, v
        ##################################
        # Project lidar points onto lwir
        lidarPointsFlir_np, jacobian_flir = cv2.projectPoints(
            lwir_lidarFlirPoints_np,
            np.eye(3, 3),
            np.zeros((3, 1)),
            self.intrinsics_dict['"flir_lwir_camera"']["K"],
            self.intrinsics_dict['"flir_lwir_camera"']["D"],
        )
        lidarPointsFlir_np = lidarPointsFlir_np.reshape((-1, 2)).astype(int)
        lidar_lwir_filter = np.logical_and(
            np.logical_and(
                0 <= lidarPointsFlir_np[:, 0],
                lidarPointsFlir_np[:, 0]
                < self.intrinsics_dict['"flir_lwir_camera"']["width"],
            ),
            np.logical_and(
                0 <= lidarPointsFlir_np[:, 1],
                lidarPointsFlir_np[:, 1]
                < self.intrinsics_dict['"flir_lwir_camera"']["height"],
            ),
        )
        lidarPointsFlir_np = lidarPointsFlir_np[lidar_lwir_filter]
        lwir_lidarFlirPoints_np = lwir_lidarFlirPoints_np[lidar_lwir_filter]
        u = lidarPointsFlir_np[:, 0]
        v = lidarPointsFlir_np[:, 1]

        lidar_depth_lwir[v, u] = lwir_lidarFlirPoints_np[:, 2]
        lidar_disparity_lwir[v, u] = np.reciprocal(lidar_depth_lwir[v, u])
        lidar_depth_lwir_mask[v, u] = True
        # self.draw_points(u, v, thermal_lwir_np, (0,255,0))
        self.draw_points(
            u,
            v,
            thermal_lwir_color_np,
            color_by_index(
                lwir_lidarFlirPoints_np, 2, min_height=0, max_height=50
            ),
        )
        del u, v
        ##################################
        # Project zed points onto mwir
        zedPointsTelops_np, jacobian_flir = cv2.projectPoints(
            mwir_zedTelopsPoints_np,
            np.eye(3, 3),
            np.zeros((3, 1)),
            self.intrinsics_dict['"telops_mwir_camera"']["K"],
            self.intrinsics_dict['"telops_mwir_camera"']["D"],
        )
        zedPointsTelops_np = zedPointsTelops_np.reshape((-1, 2)).astype(int)
        zed_mwir_filter = np.logical_and(
            np.logical_and(
                0 <= zedPointsTelops_np[:, 0],
                zedPointsTelops_np[:, 0]
                < self.intrinsics_dict['"telops_mwir_camera"']["width"],
            ),
            np.logical_and(
                0 <= zedPointsTelops_np[:, 1],
                zedPointsTelops_np[:, 1]
                < self.intrinsics_dict['"telops_mwir_camera"']["height"],
            ),
        )
        zedPointsTelops_np = zedPointsTelops_np[zed_mwir_filter]
        mwir_zedTelopsPoints_np = mwir_zedTelopsPoints_np[zed_mwir_filter]
        u = zedPointsTelops_np[:, 0]
        v = zedPointsTelops_np[:, 1]

        zed_depth_mwir[v, u] = mwir_zedTelopsPoints_np[:, 2]
        zed_disparity_mwir[v, u] = np.reciprocal(zed_depth_mwir[v, u])
        zed_depth_mwir_mask[v, u] = True
        # self.draw_points(u, v, thermal_mwir_np, (0,0,255))
        self.draw_points(
            u,
            v,
            thermal_mwir_color_np,
            color_by_index(
                mwir_zedTelopsPoints_np, 2, min_height=0, max_height=50
            ),
        )
        del u, v
        ##################################
        # Project lidar points onto mwir
        lidarPointsTelops_np, jacobian_flir = cv2.projectPoints(
            mwir_lidarTelopsPoints_np,
            np.eye(3, 3),
            np.zeros((3, 1)),
            self.intrinsics_dict['"telops_mwir_camera"']["K"],
            self.intrinsics_dict['"telops_mwir_camera"']["D"],
        )
        lidarPointsTelops_np = lidarPointsTelops_np.reshape((-1, 2)).astype(
            int
        )
        lidar_mwir_filter = np.logical_and(
            np.logical_and(
                0 <= lidarPointsTelops_np[:, 0],
                lidarPointsTelops_np[:, 0]
                < self.intrinsics_dict['"telops_mwir_camera"']["width"],
            ),
            np.logical_and(
                0 <= lidarPointsTelops_np[:, 1],
                lidarPointsTelops_np[:, 1]
                < self.intrinsics_dict['"telops_mwir_camera"']["height"],
            ),
        )
        lidarPointsTelops_np = lidarPointsTelops_np[lidar_mwir_filter]
        mwir_lidarTelopsPoints_np = mwir_lidarTelopsPoints_np[
            lidar_mwir_filter
        ]
        u = lidarPointsTelops_np[:, 0]
        v = lidarPointsTelops_np[:, 1]

        lidar_depth_mwir[v, u] = mwir_lidarTelopsPoints_np[:, 2]
        lidar_disparity_mwir[v, u] = np.reciprocal(lidar_depth_mwir[v, u])
        lidar_depth_mwir_mask[v, u] = True
        # self.draw_points(u, v, thermal_mwir_np, (0,255,0))
        self.draw_points(
            u,
            v,
            thermal_mwir_color_np,
            color_by_index(
                mwir_lidarTelopsPoints_np, 2, min_height=0, max_height=50
            ),
        )
        del u, v
        ##################################

        if flip_flag:
            depth_np = cv2.flip(depth_np, 1)
            thermal_lwir_np = cv2.flip(thermal_lwir_np, 1)
            thermal_mwir_np = cv2.flip(thermal_mwir_np, 1)

        # lwir_lidarPoints = torch.tensor(lwir_lidarPoints_np).unsqueeze(0)
        # zedPoints = torch.tensor(zedPoints_np).unsqueeze(0)

        # thermal_lwir = self.img_transform(thermal_lwir_np)
        # thermal_mwir = self.img_transform(thermal_mwir_np)

        return {
            "timestamp": frame["timestamp"],
            "thermal_lwir_np": thermal_lwir_np,
            "thermal_mwir_np": thermal_mwir_np,
            "thermal_lwir_raw_np": thermal_lwir_raw_np,
            "thermal_mwir_raw_np": thermal_mwir_raw_np,
            "zed_depth_lwir": zed_depth_lwir,
            "zed_depth_mwir": zed_depth_mwir,
            "lidar_depth_lwir": lidar_depth_lwir,
            "lidar_depth_mwir": lidar_depth_mwir,
            "zed_disparity_lwir": zed_disparity_lwir,
            "zed_disparity_mwir": zed_disparity_mwir,
            "lidar_disparity_lwir": lidar_disparity_lwir,
            "lidar_disparity_mwir": lidar_disparity_mwir,
            "zed_depth_lwir_mask": zed_depth_lwir_mask,
            "zed_depth_mwir_mask": zed_depth_mwir_mask,
            "lidar_depth_lwir_mask": lidar_depth_lwir_mask,
            "lidar_depth_mwir_mask": lidar_depth_mwir_mask,
            "thermal_lwir_color_np": thermal_lwir_color_np,
            "thermal_mwir_color_np": thermal_mwir_color_np,
            "zed2i_stereo_np": zed2i_stereo_np,
            "zed2i_left_np": zed2i_left_np,
            "zed2i_right_np": zed2i_right_np,
        }


class TVD_LWIR_RAW_LiDAR(ThermalVoyager_ThermalDepthDataset):
    """
    Thermal Voyeger Dataset
        LWIR Raw data
        LiDAR depth
    """

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        thermal_lwir_np = frame["thermal_lwir_np"]
        thermal_lwir_raw_np = frame["thermal_lwir_raw_np"]
        lidar_disparity_lwir = frame["lidar_disparity_lwir"]
        lidar_depth_lwir_mask = frame["lidar_depth_lwir_mask"]

        x = thermal_lwir_raw_np
        x_raw = thermal_lwir_np  # Image used for visualization
        y = lidar_disparity_lwir
        mask = lidar_depth_lwir_mask

        return [x, torch.tensor(x_raw).unsqueeze(0), mask, y]


class TVD_LWIR_HIST_LiDAR(ThermalVoyager_ThermalDepthDataset):
    """
    Thermal Voyeger Dataset
        LWIR Histogram equalized data
        LiDAR depth
    """

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        thermal_lwir_np = frame["thermal_lwir_np"]
        thermal_lwir_color_np = frame["thermal_lwir_color_np"]
        thermal_lwir_raw_np = frame["thermal_lwir_raw_np"]
        lidar_disparity_lwir = frame["lidar_disparity_lwir"]
        lidar_depth_lwir_mask = frame["lidar_depth_lwir_mask"]

        x = self.img_transform(thermal_lwir_np)
        x_raw = torch.tensor(thermal_lwir_color_np).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.tensor(lidar_disparity_lwir).unsqueeze(0)
        mask = torch.tensor(lidar_depth_lwir_mask).unsqueeze(0)

        return [x, x_raw, mask, y]


class TVD_MWIR_HIST_LiDAR(ThermalVoyager_ThermalDepthDataset):
    """
    Thermal Voyeger Dataset
        MWIR Histogram equalized data
        LiDAR depth
    """

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        thermal_mwir_np = frame["thermal_mwir_np"]
        thermal_mwir_color_np = frame["thermal_mwir_color_np"]
        thermal_mwir_raw_np = frame["thermal_mwir_raw_np"]
        lidar_disparity_mwir = frame["lidar_disparity_mwir"]
        lidar_depth_mwir_mask = frame["lidar_depth_mwir_mask"]

        x = self.img_transform(thermal_mwir_np)
        x_raw = torch.tensor(thermal_mwir_color_np).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.tensor(lidar_disparity_mwir).unsqueeze(0)
        mask = torch.tensor(lidar_depth_mwir_mask).unsqueeze(0)

        return [x, x_raw, mask, y]


##################################################################################
def identity_transform(x):
    return x

class ThermalVoyagerCarStateDataset(Dataset):
    def __init__(
        self,
        tvd_dataset_path="/home/shared/Thermal_Voyager/Processed/2023-04-18/1/",
        fault_delay=0.025,  # seconds
        debug=False,
        recompute_frame_mapping=False,
        recompute_car_state_mapping=True,
        autopilot_dataset_path="/home/shared/car_dataset/car_dataset/2023-04-18_22:27:36.166407/",
        timing_offset=0.0,
        transform=identity_transform,
        **kwargs
    ) -> None:
        self.debug = debug
        self.fault_delay = fault_delay
        self.img_transform = transform

        if self.debug:
            print("=" * 10)
            print("Loading ThermalVoyagerDataset")
        self.tvd_dataset = ThermalVoyagerDataset(
            dataset_path=tvd_dataset_path,
            fault_delay=fault_delay,
            recompute_frame_mapping=recompute_frame_mapping,
            debug=debug,
            **kwargs
        )

        if self.debug:
            print("=" * 10)
            print("Loading AutopilotIterCarStateTrajSteer")
        self.autopilot_dataset = (
            autopilot_carstate_iterator.AutopilotIterCarStateTrajSteer(
                autopilot_dataset_path,
                cam_left="rgb_1.mp4",
                cam_center="rgb_2.mp4",
                cam_right="rgb_3.mp4",
                car_state_csv="carState.csv",
                gps_csv="gps.csv",
                debug=debug,
            )
        )

        self.cache_path = self.tvd_dataset.cache_path
        if self.debug:
            print("cache_path", self.cache_path)

        self.dataset_timestamps = np.array(
            self.tvd_dataset.dataset_timestamps.copy()
        )
        car_state_start = self.autopilot_dataset[0][0]["timestamp"]
        car_state_end = self.autopilot_dataset[
            len(self.autopilot_dataset) - 1
        ][0]["timestamp"]
        # self.car_state_timestamps = np.array(
        #     self.autopilot_dataset.dataset.car_state[
        #         (self.autopilot_dataset.dataset.car_state["timestamp"] > car_state_start) &
        #         (self.autopilot_dataset.dataset.car_state["timestamp"] < car_state_end)
        #     ]["timestamp"].tolist()
        # ) * 10**9
        self.car_state_timestamps = self.autopilot_dataset.timestamps
        # for car_frame in tqdm(self.autopilot_dataset):
        #     timestamp = car_frame[0]["timestamp"]
        #     self.car_state_timestamps.append(timestamp)

        self.car_state_timestamps = (
            np.array(self.car_state_timestamps) * 10**9
        )

        self.car_state_mapping_csv = os.path.join(
            self.cache_path, "CarState.csv"
        )
        if (
            os.path.isfile(self.car_state_mapping_csv)
            and not recompute_car_state_mapping
        ):
            self.car_state_mapping_df = pd.read_csv(self.car_state_mapping_csv)
        else:
            self.car_state_mapping_df = {
                "dataset_timestamps": [],
                "car_state_timestamps": [],
                "valid_timestamps": [],
            }
            self.car_state_mapping_df["valid_timestamps"] = np.ones_like(
                self.dataset_timestamps, dtype=bool
            )
            for timestamp_index in tqdm(range(len(self.dataset_timestamps))):
                timestamp_ns = self.dataset_timestamps[timestamp_index]
                timestamp_ns_offset = timestamp_ns - timing_offset * 10**9
                ts = get_closest_timestamp(
                    self.car_state_timestamps,
                    timestamp_ns_offset,
                    fault_delay=float("inf"),
                )

                self.car_state_mapping_df["dataset_timestamps"] += [
                    timestamp_ns
                ]
                self.car_state_mapping_df["car_state_timestamps"] += [ts]

                if abs(timestamp_ns_offset - ts) > self.fault_delay * 10**9:
                    self.car_state_mapping_df["valid_timestamps"][
                        timestamp_index
                    ] = False

            self.car_state_mapping_df = pd.DataFrame(self.car_state_mapping_df)
            self.car_state_mapping_df.to_csv(self.car_state_mapping_csv)

        self.frames_dropped = len(
            self.car_state_mapping_df["valid_timestamps"]
        ) - np.sum(self.car_state_mapping_df["valid_timestamps"])
        if self.debug:
            print("self.frames_dropped", self.frames_dropped)

        self.dataset_timestamps = self.dataset_timestamps[
            self.car_state_mapping_df["valid_timestamps"]
        ]
        self.dataset_timestamps = self.dataset_timestamps.tolist()

        self.car_state_timestamps = self.car_state_timestamps.tolist()

    def __len__(self):
        return len(self.dataset_timestamps)

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
        timestamp_ns = self.dataset_timestamps[key]
        timestamp = timestamp_ns / 10**9

        dataset_ts = self.dataset_timestamps[key]
        car_state_ts = self.car_state_mapping_df[
            self.car_state_mapping_df["dataset_timestamps"] == dataset_ts
        ]["car_state_timestamps"].iloc[0]

        assert car_state_ts in self.car_state_timestamps
        car_state_index = self.car_state_timestamps.index(car_state_ts)

        assert dataset_ts in self.tvd_dataset.dataset_timestamps
        dataset_index = self.tvd_dataset.dataset_timestamps.index(dataset_ts)

        if self.debug:
            print(
                "car_state_index", car_state_index, len(self.autopilot_dataset)
            )
            print("dataset_index", dataset_index, len(self.tvd_dataset))

        car_state_frame = self.autopilot_dataset[car_state_index]
        dataset_frame = self.tvd_dataset[dataset_index]

        zed2i_stereo_np = cv2.imread(dataset_frame["zed2i_stereo_data_path"])
        zed2i_left_np = zed2i_stereo_np[:, : zed2i_stereo_np.shape[1] // 2, :]
        zed2i_right_np = zed2i_stereo_np[:, zed2i_stereo_np.shape[1] // 2 :, :]
        if self.debug:
            print("zed2i_stereo_np.shape", zed2i_stereo_np.shape)

        thermal_lwir_np = cv2.imread(dataset_frame["flir_lwir_data_path"])
        thermal_mwir_np = cv2.imread(dataset_frame["telops_mwir_data_path"])

        thermal_lwir_color_np = thermal_lwir_np.copy()
        thermal_mwir_color_np = thermal_mwir_np.copy()

        thermal_lwir_raw_np = cv2.imread(
            dataset_frame["flir_lwir_raw_data_path"]
        )
        thermal_mwir_raw_np = cv2.imread(
            dataset_frame["telops_mwir_raw_data_path"]
        )

        return {
            "timestamp": timestamp,
            "car_state_frame": car_state_frame,
            "dataset_frame": dataset_frame,
            "thermal_lwir_np": thermal_lwir_np,
            "thermal_mwir_np": thermal_mwir_np,
            "thermal_lwir_color_np": thermal_lwir_color_np,
            "thermal_mwir_color_np": thermal_mwir_color_np,
            "thermal_lwir_raw_np": thermal_lwir_raw_np,
            "thermal_mwir_raw_np": thermal_mwir_raw_np,
            "zed2i_stereo_np": zed2i_stereo_np,
            "zed2i_left_np": zed2i_left_np,
            "zed2i_right_np": zed2i_right_np,
        }


import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from TrajNet.datasets.autopilot_iterator import autopilot_carstate_iterator
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_traj(traj):
    traj_np = traj.cpu().view(-1, 2, 10).detach().numpy()
    x = traj_np[0, 0, :]
    y = traj_np[0, 1, :]

    x_rec = [x[0]]
    y_rec = [y[0]]

    for rec_i in range(1, len(x)):
        x_rec.append(x[rec_i] + x_rec[rec_i - 1])
        y_rec.append(y[rec_i] + y_rec[rec_i - 1])

    plt.plot(y_rec, x_rec)
    plt.xlim([-1.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()


def plot_traj_img(img, img_raw, traj, camera_intrinsics):
    img_np = img_raw.cpu().detach().numpy()

    final_frame = img_np.astype(np.uint8).copy()

    traj_np = traj.cpu().view(-1, 2, 10).detach().numpy()
    x = traj_np[0, 0, :]
    y = traj_np[0, 1, :]

    x_rec = [x[0]]
    y_rec = [y[0]]

    height = -10.0

    for rec_i in range(1, len(x)):
        x_rec.append(x[rec_i] + x_rec[rec_i - 1])
        y_rec.append(y[rec_i] + y_rec[rec_i - 1])

    for x3d, y3d in zip(x_rec, y_rec):
        z3d = height
        p3d = np.array([x3d, y3d, z3d, 1.0]).reshape((4, 1))
        p2d = camera_intrinsics @ p3d
        if p2d[2][0] != 0.0:
            px, py = round(p2d[0][0] / p2d[2][0]), round(p2d[1][0] / p2d[2][0])
            final_frame = cv2.circle(final_frame, (px, py), 5, (0, 255, 0), -1)

    plt.imshow(final_frame)
    plt.show()


def get_plot_traj_img(img, img_raw, traj, camera_intrinsics, height=1.0):
    img_np = img_raw
    if type(img_raw) == torch.Tensor:
        img_np = img_raw.cpu().detach().numpy()

    final_frame = img_np.astype(np.uint8).copy()

    traj_np = traj.cpu().view(-1, 2, 10).detach().numpy()
    x = traj_np[0, 0, :]
    y = traj_np[0, 1, :]

    x_rec = [x[0]]
    y_rec = [y[0]]

    for rec_i in range(1, len(x)):
        x_rec.append(x[rec_i] + x_rec[rec_i - 1])
        y_rec.append(y[rec_i] + y_rec[rec_i - 1])

    # for x3d, y3d in zip(x_rec, y_rec):
    for x3d, y3d in zip(y_rec, x_rec):
        z3d = height
        p3d = np.array([x3d, y3d, z3d, 1.0]).reshape((4, 1))
        p2d = camera_intrinsics @ p3d
        if p2d[2][0] != 0.0:
            px, py = round(p2d[0][0] / p2d[2][0]), round(p2d[1][0] / p2d[2][0])
            final_frame = cv2.circle(final_frame, (px, py), 5, (0, 255, 0), -1)

    return final_frame


from TrajNet.utils.trajectory import (
    apply_affine_transform_on_image_and_trajectory,
    generate_morph,
)


class ThermalVoyagerCarStateDataset_lwir_raw(ThermalVoyagerCarStateDataset):
    def __init__(self, *args, **kwargs):
        super(ThermalVoyagerCarStateDataset_lwir_raw, self).__init__(
            *args, **kwargs
        )
        self.MAX_WARP_Y = 0.0
        if "MAX_WARP_Y" in kwargs:
            self.MAX_WARP_Y = kwargs["MAX_WARP_Y"]
        self.random_flip = False  # If False, no random flip
        if "random_flip" in kwargs:
            self.random_flip = kwargs["random_flip"]

    def __len__(
        self,
    ):
        new_len = super().__len__()
        if self.random_flip:
            new_len = new_len * 2
        return new_len

    def __getitem__(self, frame_index_original):
        original_len = super().__len__()
        frame_index = frame_index_original % original_len
        augment_flag = bool(frame_index_original // original_len)

        frames_tvd_auto = super().__getitem__(frame_index)
        frames_list = frames_tvd_auto["car_state_frame"]
        frame = frames_list[0]

        # frame_merged_np = frame['car_state_frame']['frame_merged']
        # frame_center_np = frame['car_state_frame']['frame_center']
        trajectory_rel = frame["trajectory_rel"]

        frame_center_np = frames_tvd_auto["thermal_lwir_raw_np"]
        frame_merged_np = np.hstack([
            frame_center_np, frame_center_np, frame_center_np
        ])
        # frame_merged_np = frames_tvd_auto['thermal_lwir_np']
        # frame_merged_np = cv2.resize(frame_merged_np, (640, 480))

        if augment_flag:
            frame_merged_np = cv2.flip(frame_merged_np, 1)
            trajectory_rel[:, 1] = -trajectory_rel[:, 1]

        warp_y = (random.random() - 0.5) * self.MAX_WARP_Y
        M_2D = generate_morph(warp_y)  # 2D affine transform 3x3

        (
            frame_merged_np,
            trajectory_rel,
        ) = apply_affine_transform_on_image_and_trajectory(
            frame_merged_np,
            trajectory_rel,
            M_2D,
            self.autopilot_dataset.intrinsic_matrix,
        )

        frame_center = torch.tensor(frame_center_np).unsqueeze(0)
        frame_merged = torch.tensor(frame_merged_np).unsqueeze(0)

        x = torch.tensor(self.img_transform({
            'image': frame_merged_np
        })['image']).unsqueeze(0)

        y = torch.tensor(trajectory_rel).unsqueeze(0)
        y = y.reshape((1, self.autopilot_dataset.trajectory_lookahead * 2))

        return [x, frame_merged, frame_center, y]


class ThermalVoyagerCarStateDataset_lwir_norm(ThermalVoyagerCarStateDataset):
    def __init__(self, *args, **kwargs):
        super(ThermalVoyagerCarStateDataset_lwir_norm, self).__init__(
            *args, **kwargs
        )
        self.MAX_WARP_Y = 0.0
        if "MAX_WARP_Y" in kwargs:
            self.MAX_WARP_Y = kwargs["MAX_WARP_Y"]
        self.random_flip = False  # If False, no random flip
        if "random_flip" in kwargs:
            self.random_flip = kwargs["random_flip"]

    def __len__(
        self,
    ):
        new_len = super().__len__()
        if self.random_flip:
            new_len = new_len * 2
        return new_len

    def __getitem__(self, frame_index_original):
        original_len = super().__len__()
        frame_index = frame_index_original % original_len
        augment_flag = bool(frame_index_original // original_len)

        frames_tvd_auto = super().__getitem__(frame_index)
        frames_list = frames_tvd_auto["car_state_frame"]
        frame = frames_list[0]

        # frame_merged_np = frame['car_state_frame']['frame_merged']
        # frame_center_np = frame['car_state_frame']['frame_center']
        trajectory_rel = frame["trajectory_rel"]

        frame_center_np = frames_tvd_auto["thermal_lwir_color_np"]
        frame_merged_np = np.hstack([
            frame_center_np, frame_center_np, frame_center_np
        ])

        if augment_flag:
            frame_center_np = cv2.flip(frame_center_np, 1)
            trajectory_rel[:, 1] = -trajectory_rel[:, 1]

        warp_y = (random.random() - 0.5) * self.MAX_WARP_Y
        M_2D = generate_morph(warp_y)  # 2D affine transform 3x3

        (
            frame_center_np,
            trajectory_rel,
        ) = apply_affine_transform_on_image_and_trajectory(
            frame_center_np,
            trajectory_rel,
            M_2D,
            self.autopilot_dataset.intrinsic_matrix,
        )

        frame_center = torch.tensor(frame_center_np).unsqueeze(0)
        frame_merged = torch.tensor(frame_merged_np).unsqueeze(0)

        x = torch.tensor(self.img_transform({
            'image': frame_center_np
        })['image']).unsqueeze(0)

        y = torch.tensor(trajectory_rel).unsqueeze(0)
        y = y.reshape((1, self.autopilot_dataset.trajectory_lookahead * 2))

        return [x, frame_merged, frame_center, y]


def get_tvd_dataset(
    tvd_base="/home/shared/Thermal_Voyager",
    car_dataset_base="/home/shared/car_dataset/",
    tvd_class=ThermalVoyagerCarStateDataset_lwir_raw,
    transform=transforms,
):
    dataset = torch.utils.data.ConcatDataset(
        [
            tvd_class(
                tvd_dataset_path=os.path.join(
                    tvd_base, "Processed/2023-04-18/1/"
                ),
                autopilot_dataset_path=os.path.join(
                    car_dataset_base, "2023-04-18_22:27:36.166407"
                ),
                debug=False,
                timing_offset=0.000,
                transform=transform,
            ),
            tvd_class(
                tvd_dataset_path=os.path.join(
                    tvd_base, "Processed/2023-04-18/2/"
                ),
                autopilot_dataset_path=os.path.join(
                    car_dataset_base, "2023-04-18_22:31:05.159478"
                ),
                debug=False,
                timing_offset=0.000,
                transform=transform,
            ),
        ]
    )

    return dataset


##################################################################################


def normalize(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
