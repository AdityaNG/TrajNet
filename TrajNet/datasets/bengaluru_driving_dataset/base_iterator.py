"""
DatasetHelper.py
    AndroidDatasetIterator
    PandaDatasetRecorder
"""

import binascii
import itertools
import math
import os
import pathlib
import pickle
import sys
from datetime import datetime, timedelta

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from . import helper
from .dataset_constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BengaluruDepthDatasetIterator:
    def __init__(
        self,
        dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1653972957447",
        settings_doc="~/Datasets/Depth_Dataset_Bengaluru/calibration/pocoX3/calib.yaml",
    ) -> None:
        self.dataset_path = os.path.expanduser(dataset_path)
        self.dataset_id = self.dataset_path.split("/")[-1]
        self.rgb_img_folder = os.path.join(self.dataset_path, "rgb_img")
        self.depth_img_folder = os.path.join(self.dataset_path, "depth_img")
        self.seg_img_folder = os.path.join(self.dataset_path, "seg_img")
        self.csv_path = os.path.join(
            self.dataset_path, self.dataset_id + ".csv"
        )

        os.path.isdir(self.dataset_path)
        os.path.isdir(self.rgb_img_folder)
        os.path.isdir(self.depth_img_folder)
        os.path.isdir(self.seg_img_folder)
        os.path.isfile(self.csv_path)

        self.settings_doc = os.path.expanduser(settings_doc)
        with open(self.settings_doc, "r") as stream:
            try:
                self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        k1 = self.cam_settings["Camera.k1"]
        k2 = self.cam_settings["Camera.k2"]
        p1 = self.cam_settings["Camera.p1"]
        p2 = self.cam_settings["Camera.p2"]
        k3 = 0
        if "Camera.k3" in self.cam_settings:
            k3 = self.cam_settings["Camera.k3"]
        self.DistCoef = np.array([k1, k2, p1, p2, k3])
        self.intrinsic_matrix = np.array(
            [
                [
                    self.cam_settings["Camera.fx"],
                    0.0,
                    self.cam_settings["Camera.cx"],
                ],
                [
                    0.0,
                    self.cam_settings["Camera.fy"],
                    self.cam_settings["Camera.cy"],
                ],
                [0.0, 0.0, 1.0],
            ]
        )

        self.width = self.cam_settings["Camera.width"]
        self.height = self.cam_settings["Camera.height"]

        self.csv_dat = pd.read_csv(self.csv_path)

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self):
        if self.line_no >= self.__len__():
            raise StopIteration
        data = self[self.line_no]
        self.line_no += 1
        return data

    def __len__(self):
        return len(self.csv_dat)

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        csv_frame = self.csv_dat.loc[key]
        timestamp = str(int(csv_frame[1]))
        # timestamp = str(csv_frame[1])
        disparity_frame_path = os.path.join(
            self.depth_img_folder, timestamp + ".png"
        )
        seg_frame_path = os.path.join(self.seg_img_folder, timestamp + ".png")
        rgb_frame_path = os.path.join(self.rgb_img_folder, timestamp + ".png")

        assert os.path.isfile(disparity_frame_path), (
            "File missing " + disparity_frame_path
        )
        assert os.path.isfile(seg_frame_path), "File missing " + seg_frame_path
        assert os.path.isfile(rgb_frame_path), "File missing " + rgb_frame_path

        disparity_frame = cv2.imread(disparity_frame_path)
        seg_frame = cv2.imread(seg_frame_path)
        rgb_frame = cv2.imread(rgb_frame_path)

        disparity_frame = cv2.cvtColor(disparity_frame, cv2.COLOR_BGR2GRAY)

        frame = {
            "rgb_frame": rgb_frame,
            "disparity_frame": disparity_frame,
            "seg_frame": seg_frame,
            "csv_frame": csv_frame,
        }

        for key in csv_frame.keys():
            frame[key] = csv_frame[key]
        return frame


Z_OFFSET = 0.0


class BengaluruOccupancyDatasetIterator(BengaluruDepthDatasetIterator):
    def __init__(
        self,
        dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1653972957447",
        settings_doc="~/Datasets/Depth_Dataset_Bengaluru/calibration/pocoX3/calib.yaml",
        grid_size=(
            40.0,
            30.0,
            4.0,
        ),  # (128/grid_scale[0], 128/grid_scale[1], 8/grid_scale[2])
        scale=(10.0, 10.0, 10.0),  # voxels per meter
    ) -> None:
        super().__init__(dataset_path, settings_doc)

        self.scale = scale
        self.grid_size = grid_size
        self.occupancy_shape = list(
            map(
                lambda ind: int(self.grid_size[ind] * self.scale[ind]),
                range(len(self.grid_size)),
            )
        )

        self.baseline = 1.0
        self.fx = self.intrinsic_matrix[0, 0]
        self.fy = self.intrinsic_matrix[1, 1]
        self.cx = self.intrinsic_matrix[0, 2]
        self.cy = self.intrinsic_matrix[1, 2]
        self.focal_length = self.fx

        self.transformation = np.eye(4, 4)
        # self.transformation[:3,:3] = Rotation.from_euler("xyz", (-1.70000000e+02,  4.83655339e-15, -6.46097591e-14),degrees=True).as_matrix() # Great Top Down View
        # self.transformation[:3,3] = [10000.0, 0.0, 0.0]
        # self.transformation[3,:3] = [0.0, 0.0, 10.0]

    def transform_occupancy_grid_to_points(
        self, occupancy_grid, threshold=0.5, device=device, skip=3
    ):
        occupancy_grid = occupancy_grid.squeeze()
        # occupancy_grid = torch.tensor(occupancy_grid, device=device)
        def f(xi):
            i, j, k = xi
            x, y, z = [
                (i) * self.grid_size[0] / (self.occupancy_shape[0] / 2),
                (j - self.occupancy_shape[1] / 2)
                * self.grid_size[1]
                / (self.occupancy_shape[1] / 2),
                (k - self.occupancy_shape[2] / 2)
                * self.grid_size[2]
                / (self.occupancy_shape[2] / 2),
            ]
            if occupancy_grid[i, j, k] > threshold:
                z = z - Z_OFFSET
                z, x, y = x, y, z
                return (x, y, z)
            return (0, 0, 0)

        # np.array([f(xi) for xi in x])
        final_points = np.array(
            [
                f(xi)
                for xi in itertools.product(
                    range(0, occupancy_grid.shape[0], skip),
                    range(0, occupancy_grid.shape[1], skip),
                    range(0, occupancy_grid.shape[2], skip),
                )
            ]
        )
        # final_points = np.fromfunction(lambda xi: f(xi), np.indices(occupancy_grid.shape))

        final_points = final_points[
            np.logical_not(
                np.logical_and(
                    final_points[:, 0] == 0,
                    final_points[:, 1] == 0,
                    final_points[:, 2] == 0,
                )
            )
        ]

        # final_points = final_points.cpu().detach().numpy()
        final_points = np.array(final_points, dtype=np.float32)
        return final_points

    def transform_points_to_occupancy_grid(self, velodyine_points_orig):
        occupancy_grid = np.zeros(self.occupancy_shape, dtype=np.float32)

        velodyine_points = velodyine_points_orig.copy()

        velodyine_points_camera = []

        # velodyine_points = np.delete(velodyine_points, 3, axis=0)

        for index in range(velodyine_points.shape[0]):
            # x, y, z = velodyine_points[:, index]
            x, y, z = velodyine_points[index, :]

            # x, y, z = x, z, y # Half
            # x, y, z = y, x, z # N
            # x, y, z = y, z, x # N
            # x, y, z = z, x, -y # Inverted
            # x, y, z = z, y, x

            # x, y, z = -y, x, z
            x, y, z = z, x, y  # Inverted
            z = z + Z_OFFSET

            if (
                np.isinf(x)
                or np.isinf(y)
                or np.isinf(z)
                or np.isnan(x)
                or np.isnan(y)
                or np.isnan(z)
            ):
                continue

            i, j, k = [
                int((x * self.occupancy_shape[0] // 2) // self.grid_size[0])
                * 2,
                int(
                    (y * self.occupancy_shape[1] // 2) // self.grid_size[1]
                    + self.occupancy_shape[1] // 2
                ),
                int(
                    (z * self.occupancy_shape[2] // 2) // self.grid_size[2]
                    + self.occupancy_shape[2] // 2
                ),
            ]

            if (
                0 < i < self.occupancy_shape[0]
                and 0 < j < self.occupancy_shape[1]
                and 0 < k < self.occupancy_shape[2]
            ):
                velodyine_points_camera.append((x, y, z))
                occupancy_grid[i, j, k] = 1.0

        velodyine_points_camera = np.array(
            velodyine_points_camera, dtype=np.float32
        )

        return {
            "occupancy_grid": occupancy_grid,
            "velodyine_points_camera": velodyine_points_camera,
        }

    def __getitem__(self, key):
        frame = super().__getitem__(key)
        disparity = frame["disparity_frame"].astype(np.float32)
        rgb_frame = cv2.cvtColor(frame["rgb_frame"], cv2.COLOR_BGR2RGB)

        depth = self.baseline * self.focal_length * np.reciprocal(disparity)
        depth[np.isinf(depth)] = self.baseline * self.focal_length
        depth[np.isnan(depth)] = self.baseline * self.focal_length
        # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        depth = depth.astype(np.float32)

        # print('depth.dtype', depth.dtype, np.max(depth), np.min(depth))
        # print('disparity.dtype', disparity.dtype, np.max(disparity), np.min(disparity))

        # depth[depth>1250.6] = float('inf')
        # depth[depth>50.0] = float('inf')
        # depth[depth>100.0] = float('inf')
        # depth[
        #     :,
        #     0:depth.shape[1]//2
        # ] = float('inf')
        hide_mask = np.zeros((self.height, self.width), dtype=bool)
        hide_mask[
            0 : depth.shape[0] // 2 :,
        ] = True
        depth[hide_mask] = float("inf")

        U, V = np.ix_(
            np.arange(self.height), np.arange(self.width)
        )  # pylint: disable=unbalanced-tuple-unpacking
        Z = depth.copy()

        X = (V - self.cx) * Z / self.fx
        Y = (U - self.cy) * Z / self.fy

        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()

        points = np.array([X, Y, Z]).T
        # points = np.array([Y,Z,X]).T

        B, G, R = rgb_frame[:, :, 0], rgb_frame[:, :, 1], rgb_frame[:, :, 2]
        B = B.flatten()
        G = G.flatten()
        R = R.flatten()

        # points_colors = np.array([R,G,B]).T /255.0
        points_colors = np.array([B, G, R]).T / 255.0

        # print('X.shape', X.shape)
        # print('points.shape', points.shape)
        # print('points_colors.shape', points_colors.shape)
        # print('points.shape', points.shape)

        occupancy_grid_data = self.transform_points_to_occupancy_grid(points)

        frame["disparity"] = disparity
        frame["depth"] = depth
        frame["points"] = points
        frame["points_colors"] = points_colors
        frame["occupancy_grid"] = occupancy_grid_data["occupancy_grid"]
        frame["velodyine_points_camera"] = occupancy_grid_data[
            "velodyine_points_camera"
        ]

        return frame


class PandaDatasetIterator:

    """
    PandaDatasetIterator
    """

    def __init__(self, csv_path, dbc_interp, invalidate_cache=False) -> None:
        print("Init path:", csv_path)
        assert type(dbc_interp) == DBCInterpreter
        self.dbc_interp = dbc_interp
        self.csv_path = csv_path
        self.folder_path = os.path.dirname(csv_path)
        cached_csv_folder = os.path.join(self.folder_path, PANDA_CACHE_DIR)
        os.makedirs(cached_csv_folder, exist_ok=True)
        self.cached_csv_path = os.path.join(
            cached_csv_folder, os.path.basename(csv_path)
        )

        if not os.path.exists(self.cached_csv_path) or invalidate_cache:
            print("Generating Cache: ", self.cached_csv_path)
            self.csv_dat = pd.read_csv(self.csv_path)
            self.csv_dat = self.csv_dat.sort_values("timestamp")
            addresses = list(map(str, self.csv_dat["address"].unique()))
            columns = [
                "timestamp",
            ] + addresses
            reformatted_data = []
            # reformatted_data.append(columns)

            for index, row in tqdm(
                self.csv_dat.iterrows(), total=self.csv_dat.shape[0]
            ):
                # if reformatted_data[-1][0] == row['timestamp']:
                if (
                    len(reformatted_data) > 1
                    and row["timestamp"] - reformatted_data[-1][0] < 0.001
                ):
                    reformatted_data[-1][
                        addresses.index(str(row["address"])) + 1
                    ] = (row["d1"], row["dddat"], row["d2"])
                else:
                    data_points = [row["timestamp"],] + [
                        None,
                    ] * len(addresses)
                    if len(reformatted_data) > 1:
                        data_points = [row["timestamp"],] + reformatted_data[
                            -1
                        ][1:]
                    data_points[addresses.index(str(row["address"])) + 1] = (
                        row["d1"],
                        row["dddat"],
                        row["d2"],
                    )
                    reformatted_data.append(data_points)

            reformatted_data = pd.DataFrame(reformatted_data)
            # reformatted_data.columns = ['index', ] + columns
            reformatted_data.columns = columns
            reformatted_data.set_index("timestamp")
            reformatted_data = reformatted_data.sort_values("timestamp")
            reformatted_data.to_csv(self.cached_csv_path, index=False)

        self.csv_dat = pd.read_csv(self.cached_csv_path)
        self.csv_dat.set_index("timestamp")

        self.start_time_csv = min(self.csv_dat["timestamp"])
        self.end_time_csv = max(self.csv_dat["timestamp"])

        self.duration_sec = self.end_time_csv - self.start_time_csv
        self.frame_count = self.csv_dat.shape[0]
        self.fps = self.frame_count / self.duration_sec
        self.line_no = 0

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self):
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __len__(self):
        return len(self.csv_dat)

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        timestamp = self.csv_dat.loc[key][0]
        return self.csv_dat.loc[key]

    def get_item_by_timestamp(self, timestamp, fault_delay=1.0):
        """
        Return frame closest to given timestamp
        Raise exception if delta between timestamp and frame is greaterthan fault_delay
        """
        closest_frames = self.get_item_between_timestamp(
            timestamp - fault_delay,
            timestamp + fault_delay,
            fault_delay=float("inf"),
        )
        closest_frames = closest_frames.reset_index(drop=True)
        closest_frame = closest_frames.iloc[
            (closest_frames["timestamp"] - timestamp).abs().argsort()[0]
        ]
        closest_ts = closest_frame["timestamp"]
        if abs(timestamp - closest_ts) > fault_delay:
            raise Exception(
                "No such timestamp, fault delay exceeded: abs(timestamp - closest_ts)="
                + str(abs(timestamp - closest_ts))
            )
        return closest_frame

    def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=0.5):
        """
        Return frame between two given timestamps
        Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
        Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
        """
        ts_dat = self.csv_dat[
            self.csv_dat["timestamp"].between(start_ts, end_ts)
        ]
        minimum_ts = min(ts_dat["timestamp"])
        if abs(minimum_ts - start_ts) > fault_delay:
            raise Exception(
                "start_ts is out of bounds: abs(minimum_ts - start_ts) > fault_delay"
            )
        maximum_ts = max(ts_dat["timestamp"])
        if abs(maximum_ts - end_ts) > fault_delay:
            raise Exception(
                "end_ts is out of bounds: abs(maximum_ts - end_ts) > fault_delay"
            )
        return ts_dat

    def __str__(self) -> str:
        res = "----------------------------------------------------" + "\n"
        res += "PandaDatasetIterator('" + self.csv_path + "')" + "\n"
        res += "----------------------------------------------------" + "\n"
        res += "self.fps:        \t" + str(self.fps) + "\n"
        res += "self.frame_count:\t" + str(self.frame_count) + "\n"
        res += (
            "self.start_time_csv:\t"
            + str(datetime.fromtimestamp(self.start_time_csv))
            + "\n"
        )
        res += (
            "self.end_time_csv:\t"
            + str(datetime.fromtimestamp(self.end_time_csv))
            + "\n"
        )
        res += (
            "self.duration:    \t"
            + str(timedelta(seconds=self.duration_sec))
            + "\n"
        )
        res += "----------------------------------------------------"
        return res

    def __repr__(self) -> str:
        return str(self)


class AndroidDatasetIterator:

    """
    AndroidDatasetIterator
    Iterates through dataset, given the folder_path
    """

    def __init__(self, folder_path=DATASET_LIST[-1], scale_factor=1.0) -> None:
        print("Init path:", folder_path)
        self.folder_path = folder_path
        self.scale_factor = scale_factor
        self.old_frame_number = 0
        self.line_no = 0

        self.id = folder_path.split("/")[-1]
        # self.start_time = int(self.id)
        self.csv_path = os.path.join(folder_path, self.id + ".csv")
        self.mp4_path = os.path.join(folder_path, self.id + ".mp4")
        self.depth_mp4_path = os.path.join(
            folder_path, "depth_" + self.id + ".mp4"
        )

        # CSV stores time in ms
        self.csv_dat = pd.read_csv(self.csv_path)
        self.start_time = self.csv_dat["Timestamp"].iloc[0]
        self.csv_dat = self.csv_dat.sort_values("Timestamp")

        self.cap = cv2.VideoCapture(self.mp4_path)

        # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Computed video duration from FPS and number of video frames
        self.duration = self.frame_count / self.fps

        self.start_time_csv = min(self.csv_dat["Timestamp"])
        self.end_time_csv = max(self.csv_dat["Timestamp"])
        # Computed Duration the CSV file runs for
        self.expected_duration = (
            self.end_time_csv - self.start_time_csv
        ) / 1000.0

        self.csv_fps = len(self.csv_dat) / self.expected_duration

        # Expected FPS from CSV duration and number of frames
        self.expected_fps = self.frame_count / self.expected_duration
        # TODO: Perform Plausibility check on self.expected_fps and self.fps

    def __len__(self):
        return len(self.csv_dat)

    def __getitem__(self, key):
        if key > len(self):
            raise IndexError("Out of bounds; key=", key)
        timestamp = self.csv_dat.loc[key][0]
        time_from_start = timestamp - self.start_time_csv
        frame_number = round(time_from_start * self.fps / 1000)

        delta = abs(frame_number - self.old_frame_number)
        if frame_number >= self.old_frame_number and delta < 5:
            for _ in range(delta - 1):
                ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            print("cap.set: ", delta)

        self.old_frame_number = frame_number

        ret, frame = self.cap.read()
        if ret:
            w = int(frame.shape[1] * self.scale_factor)
            h = int(frame.shape[0] * self.scale_factor)
            final_frame = cv2.resize(frame, (w, h))
            return self.csv_dat.loc[key], final_frame

        raise IndexError(
            "Frame number not catured: ", frame_number, ", key=", key
        )

    def generate_depth_map(self):
        if not os.path.exists(self.depth_mp4_path):
            # TODO: Generate Depth Map data
            pass

    def get_item_by_timestamp(self, timestamp, fault_delay=1000):
        """
        Return frame closest to given timestamp
        Raise exception if delta between timestamp and frame is greaterthan fault_delay
        """
        closest_frames = self.get_item_between_timestamp(
            timestamp - fault_delay,
            timestamp + fault_delay,
            fault_delay=float("inf"),
        )
        closest_frames = closest_frames.reset_index(drop=True)
        closest_frame = closest_frames.iloc[
            (closest_frames["Timestamp"] - timestamp).abs().argsort()[0]
        ]
        closest_ts = closest_frame["Timestamp"]
        if abs(timestamp - closest_ts) > fault_delay:
            raise Exception(
                "No such timestamp, fault delay exceeded: abs(timestamp - closest_ts)="
                + str(abs(timestamp - closest_ts))
            )

        closest_ts_index = self.csv_dat.index[
            self.csv_dat["Timestamp"] == closest_ts
        ].tolist()[0]
        return self.__getitem__(closest_ts_index)
        # return closest_frame

    def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=500):
        """
        Return frame between two given timestamps
        Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
        Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
        """
        ts_dat = self.csv_dat[
            self.csv_dat["Timestamp"].between(start_ts, end_ts)
        ]
        if len(ts_dat) == 0:
            raise Exception("No such timestamp")
        minimum_ts = min(ts_dat["Timestamp"])  # / 1000.0
        if abs(minimum_ts - start_ts) > fault_delay:
            raise Exception(
                "start_ts is out of bounds: abs(minimum_ts - start_ts)="
                + str(abs(minimum_ts - start_ts))
            )
        maximum_ts = max(ts_dat["Timestamp"])  # / 1000.0
        if abs(maximum_ts - end_ts) > fault_delay:
            raise Exception(
                "end_ts is out of bounds: abs(minimum_ts - start_ts)="
                + str(abs(maximum_ts - end_ts))
            )
        return ts_dat

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self):
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __str__(self) -> str:
        res = "----------------------------------------------------" + "\n"
        res += "AndroidDatasetIterator('" + self.folder_path + "')" + "\n"
        res += "----------------------------------------------------" + "\n"
        res += "self.fps:        \t" + str(self.fps) + "\n"
        res += "self.frame_count:\t" + str(self.frame_count) + "\n"
        res += (
            "self.start_time_csv:\t"
            + str(datetime.fromtimestamp(self.start_time_csv / 1000))
            + "\n"
        )
        res += (
            "self.end_time_csv:\t"
            + str(datetime.fromtimestamp(self.end_time_csv / 1000))
            + "\n"
        )
        res += (
            "self.expected_duration:\t"
            + str(timedelta(seconds=self.expected_duration))
            + "\n"
        )
        res += "self.expected_fps:\t" + str(self.expected_fps) + "\n"
        res += "self.csv_fps:        \t" + str(self.csv_fps) + "\n"
        res += "----------------------------------------------------"
        return res

    def __repr__(self) -> str:
        return str(self)

    def __del__(self):
        pass


class MergedDatasetIterator:

    """
    MergedDatasetIterator
    Iterates through dataset, given a AndroidDatasetIterator and a PandaDatasetIterator
    """

    def __init__(
        self,
        phone_iter: AndroidDatasetIterator,
        panda_iter: PandaDatasetIterator,
        settings_doc="~/Datasets/Depth_Dataset_Bengaluru/calibration/pocoX3/calib.yaml",
        compute_trajectory=False,
        invalidate_cache=False,
        start_index=None,
        stop_index=None,
        step_indices=None,
    ) -> None:
        assert type(phone_iter) == AndroidDatasetIterator
        assert type(panda_iter) == PandaDatasetIterator
        # assert (start_index==None and stop_index==None and step_indices==None) or (start_index!=None and stop_index!=None and step_indices!=None), "All start, end and skip indices must be provided or none"
        # TODO: Generate ID for this pair of phone_iter and panda_iter

        self.compute_trajectory = compute_trajectory
        self.phone_iter = phone_iter
        self.panda_iter = panda_iter
        self.group = [
            (
                self.phone_iter.start_time_csv / 1000.0,
                self.phone_iter.end_time_csv / 1000.0,
            ),
            (self.panda_iter.start_time_csv, self.panda_iter.end_time_csv),
        ]
        self.start_time, self.end_time = helper.intersection_of_group(
            self.group
        )

        self.duration = self.end_time - self.start_time
        self.IOU = helper.IOU_of_group(self.group)

        self.phone_dat = self.phone_iter.get_item_between_timestamp(
            self.start_time * 1000.0, self.end_time * 1000.0
        )
        # self.phone_dat = self.phone_iter.get_item_between_timestamp(self.start_time, self.end_time)
        self.phone_frame_count = len(self.phone_dat)
        self.phone_fps = self.phone_frame_count / self.duration

        self.panda_dat = self.panda_iter.get_item_between_timestamp(
            self.start_time, self.end_time
        )
        self.panda_frame_count = len(self.panda_dat)
        self.panda_fps = self.panda_frame_count / self.duration

        self.frame_count_original = max(
            self.phone_frame_count, self.panda_frame_count
        )

        if start_index == None:
            start_index = 0
        if stop_index == None:
            stop_index = self.frame_count_original
        if step_indices == None:
            step_indices = 1
        assert type(start_index) == int
        assert type(stop_index) == int
        assert type(step_indices) == int
        assert start_index >= 0
        assert stop_index <= self.frame_count_original
        self.start_index = start_index
        self.stop_index = stop_index
        self.step_indices = step_indices

        self.frame_count = self.frame_count_original
        self.fps = max(self.phone_fps, self.panda_fps)

        self.settings_doc = settings_doc
        with open(self.settings_doc, "r") as stream:
            try:
                self.cam_settings = yaml.load(stream, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)
        k1 = self.cam_settings["Camera.k1"]
        k2 = self.cam_settings["Camera.k2"]
        p1 = self.cam_settings["Camera.p1"]
        p2 = self.cam_settings["Camera.p2"]
        k3 = 0
        if "Camera.k3" in self.cam_settings:
            k3 = self.cam_settings["Camera.k3"]
        self.DistCoef = np.array([k1, k2, p1, p2, k3])
        self.camera_matrix = np.array(
            [
                [
                    self.cam_settings["Camera.fx"],
                    0.0,
                    self.cam_settings["Camera.cx"],
                ],
                [
                    0.0,
                    self.cam_settings["Camera.fy"],
                    self.cam_settings["Camera.cy"],
                ],
                [0.0, 0.0, 1.0],
            ]
        )

        self.folder_path = os.path.dirname(self.phone_iter.csv_path)
        cached_trajectory_folder = os.path.join(
            self.folder_path, TRAJECTORY_CACHE_DIR
        )
        os.makedirs(cached_trajectory_folder, exist_ok=True)
        self.cached_trajectory_path = os.path.join(
            cached_trajectory_folder,
            os.path.basename(self.phone_iter.csv_path) + ".pkl",
        )
        if self.compute_trajectory:
            if (
                not os.path.exists(self.cached_trajectory_path)
                or invalidate_cache
            ):
                self.compute_slam()
            else:
                print(
                    "Loading trajectory from cache: ",
                    self.cached_trajectory_path,
                )
                with open(self.cached_trajectory_path, "rb") as handle:
                    self.trajectory = pickle.load(handle)

                # self.trajectory = pd.read_csv(self.cached_trajectory_path)
        else:
            self.trajectory = pd.DataFrame(
                {"x": [], "y": [], "z": [], "rot": []}
            )

        if (
            self.start_index != 0
            or self.step_indices != self.frame_count_original
            or self.step_indices != 1
        ):
            # Recompute
            self.start_time, self.end_time = (
                self.start_time + self.start_index / self.fps,
                self.start_time + self.stop_index / self.fps,
            )
            self.duration = self.end_time - self.start_time

            self.phone_dat = self.phone_iter.get_item_between_timestamp(
                self.start_time * 1000.0, self.end_time * 1000.0
            )
            self.phone_dat = self.phone_dat[
                self.phone_dat.index % self.step_indices == 0
            ]
            self.phone_frame_count = len(self.phone_dat)
            self.phone_fps = self.phone_frame_count / self.duration

            self.panda_dat = self.panda_iter.get_item_between_timestamp(
                self.start_time, self.end_time
            )
            self.panda_dat = self.panda_dat[
                self.panda_dat.index % self.step_indices == 0
            ]
            self.panda_frame_count = len(self.panda_dat)
            self.panda_fps = self.panda_frame_count / self.duration

            self.frame_count = max(
                self.phone_frame_count, self.panda_frame_count
            )
            self.fps = max(self.phone_fps, self.panda_fps)

    def compute_slam(
        self,
        scale_factor=0.25,
        enable_plot=False,
        plot_3D_x=250,
        plot_3D_y=500,
    ):
        sys.path.append(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(), "extras/pyslam"
            )
        )
        # from extras.pyslam.visual_odometry import VisualOdometry
        from camera import PinholeCamera
        from feature_tracker import feature_tracker_factory
        from feature_tracker_configs import FeatureTrackerConfigs
        from visual_imu_gps_odometry import Visual_IMU_GPS_Odometry
        from visual_odometry import VisualOdometry

        self.trajectory = {"x": [], "y": [], "z": [], "rot": []}

        cam = PinholeCamera(
            self.cam_settings["Camera.width"] * scale_factor,
            self.cam_settings["Camera.height"] * scale_factor,
            self.cam_settings["Camera.fx"] * scale_factor,
            self.cam_settings["Camera.fy"] * scale_factor,
            self.cam_settings["Camera.cx"] * scale_factor,
            self.cam_settings["Camera.cy"] * scale_factor,
            self.DistCoef,
            self.cam_settings["Camera.fps"],
        )
        num_features = (
            2000  # how many features do you want to detect and track?
        )

        # select your tracker configuration (see the file feature_tracker_configs.py)
        # LK_SHI_TOMASI, LK_FAST
        # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
        tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
        tracker_config["num_features"] = num_features

        feature_tracker = feature_tracker_factory(**tracker_config)
        print(feature_tracker)
        # create visual odometry object
        self.vo = Visual_IMU_GPS_Odometry(cam, None, feature_tracker)
        print("Computing Trajectory")
        plot_3D = np.zeros((plot_3D_x, plot_3D_y, 3))
        for img_id in tqdm(range(0, self.frame_count, 1)):
            data_frame = self.__getitem__(img_id)

            phone_data_frame, phone_img_frame = data_frame["phone_frame"]
            panda_data_frame = data_frame["panda_frame"]

            phone_img_frame_scaled = cv2.resize(
                phone_img_frame, (0, 0), fx=scale_factor, fy=scale_factor
            )

            self.vo.track(
                phone_img_frame_scaled,
                img_id,
                accel_data=np.array(
                    [
                        phone_data_frame["linear_acc_x"],
                        phone_data_frame["linear_acc_y"],
                        phone_data_frame["linear_acc_z"],
                    ]
                ).reshape((3, 1)),
                gyro_data=np.array(
                    [
                        phone_data_frame["RotationV X"],
                        phone_data_frame["RotationV Y"],
                        phone_data_frame["RotationV Z"],
                        phone_data_frame["RotationV W"],
                        phone_data_frame["RotationV Acc"],
                    ]
                ),
                gps_data=np.array(
                    [
                        phone_data_frame["Longitude"],
                        phone_data_frame["Latitude"],
                        phone_data_frame["speed"],
                        phone_data_frame["heading"],
                    ]
                ),
                timestamp=phone_data_frame["Timestamp"],
            )
            if img_id > 2:
                x, y, z = self.vo.traj3d_est[-1]
                rot = np.array(self.vo.cur_R, copy=True)
            else:
                # x, y, z = [0.0], [0.0], [0.0]
                x, y, z = 0.0, 0.0, 0.0
                rot = np.eye(3, 3)

            if type(x) != float:
                x = float(x[0])
            if type(y) != float:
                y = float(y[0])
            if type(z) != float:
                z = float(z[0])

            self.trajectory["x"] += [x]
            self.trajectory["y"] += [y]
            self.trajectory["z"] += [z]
            self.trajectory["rot"] += [rot]

            if enable_plot:
                p3x = int(x / 10 + plot_3D_x // 2)
                p3y = int(z / 10 + plot_3D_y // 2)
                if p3x in range(0, plot_3D_x) and p3y in range(0, plot_3D_y):
                    plot_3D = cv2.circle(
                        plot_3D, (p3y, p3x), 2, (0, 255, 0), 1
                    )

            if enable_plot:
                cv2.imshow("plot_3D", plot_3D)
                cv2.imshow("Camera", self.vo.draw_img)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break

        self.trajectory = pd.DataFrame(self.trajectory)
        # self.trajectory.to_csv(self.cached_trajectory_p]ath, index=False)
        with open(self.cached_trajectory_path, "wb") as handle:
            pickle.dump(
                self.trajectory, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, key):
        if type(key) == int:
            key_original = key
            if key > len(self):
                raise IndexError("Out of bounds; key=", key)

            # key = key * self.step_indices

            frame_ts = (
                self.start_time + key / self.fps
            )  # frame timestamp in seconds
            panda_frame = self.panda_iter.get_item_by_timestamp(frame_ts)
            phone_frame = self.phone_iter.get_item_by_timestamp(
                frame_ts * 1000.0
            )
            # phone_frame = self.phone_iter[key]

            return {
                "panda_frame": panda_frame,
                "phone_frame": phone_frame,
            }
        elif type(key) == slice:
            return MergedDatasetIterator(
                phone_iter=self.phone_iter,
                panda_iter=self.panda_iter,
                settings_doc=self.settings_doc,
                compute_trajectory=self.compute_trajectory,
                invalidate_cache=False,
                start_index=key.start,
                stop_index=key.stop,
                step_indices=key.step,
            )
        else:
            raise IndexError(
                "Unknown key type; key="
                + str(key)
                + ", type(key)="
                + str(type(key))
            )

    def __iter__(self):
        self.line_no = 0
        return self

    def __next__(self):
        if self.line_no > self.__len__():
            raise StopIteration
        data = self.__getitem__(self.line_no)
        self.line_no += 1
        return data

    def __str__(self) -> str:
        res = "----------------------------------------------------" + "\n"
        res += "MergedDatasetIterator" + "\n"
        res += "----------------------------------------------------" + "\n"
        res += (
            "self.start_time:\t"
            + str(datetime.fromtimestamp(self.start_time))
            + "\n"
        )
        res += (
            "self.end_time:\t\t"
            + str(datetime.fromtimestamp(self.end_time))
            + "\n"
        )
        res += (
            "self.duration:\t\t" + str(timedelta(seconds=self.duration)) + "\n"
        )
        res += "self.IOU:\t\t" + str(round(self.IOU * 100, 2)) + " %" + "\n"
        res += "self.frame_count:\t" + str(self.frame_count) + "\n"
        res += "self.fps:\t\t" + str(self.fps) + "\n"
        res += "----------------------------------------------------"
        return res

    def get_item_by_timestamp(self, timestamp, fault_delay=1):
        """
        Return frame closest to given timestamp
        Raise exception if delta between timestamp and frame is greaterthan fault_delay
        """
        closest_frames = self.get_item_between_timestamp(
            timestamp - fault_delay,
            timestamp + fault_delay,
            fault_delay=float("inf"),
        )
        closest_frames["panda_frame"] = closest_frames[
            "panda_frame"
        ].reset_index(drop=True)
        closest_frames["phone_frame"] = closest_frames[
            "phone_frame"
        ].reset_index(drop=True)
        closest_frame = {
            "panda_frame": closest_frames["panda_frame"].iloc[
                (closest_frames["panda_frame"]["timestamp"] - timestamp)
                .abs()
                .argsort()[0]
            ],
            "phone_frame": closest_frames["phone_frame"].iloc[
                (closest_frames["phone_frame"]["timestamp"] - timestamp)
                .abs()
                .argsort()[0]
            ],
        }
        panda_closest_ts = closest_frame["panda_frame"]["timestamp"]
        phone_closest_ts = closest_frame["phone_frame"]["timestamp"]

        if abs(panda_closest_ts - phone_closest_ts) > fault_delay:
            raise Exception(
                "Phone Panda delta too large"
                + str(abs(panda_closest_ts - phone_closest_ts))
            )
        return closest_frame

    def get_item_between_timestamp(self, start_ts, end_ts, fault_delay=0.5):
        """
        Return frame between two given timestamps
        Raise exception if delta between start_ts and minimum_ts is greater than fault_delay
        Raise exception if delta between end_ts and maximum_ts is greater than fault_delay
        """
        start_frame_ts = (
            self.start_time + start_ts / self.fps
        )  # frame timestamp in seconds
        end_frame_ts = (
            self.start_time + end_frame_ts / self.fps
        )  # frame timestamp in seconds
        panda_frame = self.panda_iter.get_item_between_timestamp(
            start_frame_ts, end_frame_ts
        )
        phone_frame = self.phone_iter.get_item_by_timestamp(
            start_frame_ts * 1000.0, end_frame_ts * 1000
        )
        return {
            "panda_frame": panda_frame,
            "phone_frame": phone_frame,
        }

    def __repr__(self) -> str:
        return str(self)

    def __del__(self):
        pass


class DBCInterpreter:

    """
    DBCInterpreter
    Interprets DBC file, given a DBC file path
    """

    def __init__(self, dbc_path="dbc/honda_city.dbc") -> None:
        import cantools

        print("dbc_path:", dbc_path)
        self.dbc_path = dbc_path
        self.db = cantools.database.load_file(self.dbc_path)
        self.id_to_datatype = {}

    def interpret_can_frame(self, data):
        # Given a CAN DataFrame from PandaCSVInterpreter, produce output in dict
        # For example, result = {'throttle':30, 'rpm': 1200, ....}
        result = {}
        for key in data.keys():
            if key != "timestamp":
                try:
                    if type(data[key]) == str:
                        tuple_dat = eval(data[key])
                        hex_frame = binascii.unhexlify(eval(tuple_dat[1]))
                        new_dict = self.db.decode_message(int(key), hex_frame)
                        for k in new_dict:
                            result[k] = new_dict[k]
                except KeyError:
                    pass
                except ValueError:
                    pass

        return result

    def __getitem__(self, key):
        return self.id_to_datatype[key]

    def __len__(self) -> int:
        return len(self.id_to_datatype)

    def __str__(self) -> str:
        res = "----------------------------------------------------" + "\n"
        res += "DBCInterpreter('" + self.dbc_path + "')" + "\n"
        res += "----------------------------------------------------" + "\n"
        for k in self.id_to_datatype:  # Print out mapping from ID to data
            res += str(k) + ":\t" + str(self.id_to_datatype[k]) + "\n"
        res += "----------------------------------------------------"
        return res

    def __repr__(self) -> str:
        return str(self)

    def __del__(self):
        pass


def compute_depth(midas, rgb_frame, transform, device):
    import torch

    input_batch = transform(rgb_frame).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()

    return output


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates using the Haversine formula.
    """
    R = 6371  # Earth radius in kilometers
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # Convert to meters


def main_grid(point_cloud_array, plot2D, plot3D):
    depth_dataset = BengaluruOccupancyDatasetIterator(
        dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1658384924059"
    )
    scale = 0.3

    if plot2D:
        plt.ion()

    for frame in depth_dataset:
        disparity_frame = frame["disparity_frame"]
        disparity_frame_rgb = cv2.applyColorMap(
            disparity_frame, cv2.COLORMAP_PLASMA
        )

        occupancy_grid = frame["occupancy_grid"]
        occupancy_grid_torch = torch.tensor(occupancy_grid).unsqueeze(0)
        print("occupancy_grid_torch.shape", occupancy_grid_torch.shape)
        grid_points = depth_dataset.transform_occupancy_grid_to_points(
            occupancy_grid_torch
        )

        rgb_frame = frame["rgb_frame"]

        frame_vis = np.concatenate(
            [
                rgb_frame,
                disparity_frame_rgb,
            ],
            0,
        )
        frame_vis = cv2.resize(frame_vis, (0, 0), fx=scale, fy=scale)
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)

        if plot3D:
            point_cloud_array.put(
                generate_3D_frame(grid_points, np.ones_like(grid_points))
            )

        if plot2D:
            plt.imshow(frame_vis)
            plt.pause(0.01)
            plt.show()

            # Check if key is pressed
            if plt.waitforbuttonpress(0.001):
                print("Key pressed, exiting...")
                exit()


def main_depth_seg(point_cloud_array, plot2D, plot3D):
    depth_dataset = BengaluruDepthDatasetIterator(
        dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1658384924059"
    )
    scale = 0.3

    if plot2D:
        plt.ion()

    for frame in depth_dataset:
        disparity_frame = frame["disparity_frame"]
        seg_frame = frame["seg_frame"]
        disparity_frame_rgb = cv2.applyColorMap(
            disparity_frame, cv2.COLORMAP_PLASMA
        )

        rgb_frame = frame["rgb_frame"]

        frame_vis = np.concatenate(
            [rgb_frame, disparity_frame_rgb, seg_frame], 0
        )
        frame_vis = cv2.resize(frame_vis, (0, 0), fx=scale, fy=scale)
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)

        if plot2D:
            plt.imshow(frame_vis)
            plt.pause(0.01)
            plt.show()

            # Check if key is pressed
            if plt.waitforbuttonpress(0.001):
                print("Key pressed, exiting...")
                exit()


def main_depth(point_cloud_array, plot2D, plot3D):
    import torch

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    depth_dataset = BengaluruOccupancyDatasetIterator(
        dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1658384924059"
    )
    scale = 0.3

    model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    midas.to(device)
    midas.eval()

    if plot2D:
        plt.ion()

    for frame in depth_dataset:
        disparity_frame = frame["disparity_frame"]
        disparity_frame_rgb = cv2.applyColorMap(
            disparity_frame, cv2.COLORMAP_PLASMA
        )

        occupancy_grid = frame["occupancy_grid"]
        occupancy_grid_torch = torch.tensor(occupancy_grid).unsqueeze(0)
        grid_points = depth_dataset.transform_occupancy_grid_to_points(
            occupancy_grid_torch
        )

        rgb_frame = frame["rgb_frame"]

        depth_pred = compute_depth(midas, rgb_frame, transform, device)
        depth_pred = (depth_pred - np.min(depth_pred)) / (
            np.max(depth_pred) - np.min(depth_pred)
        )
        depth_pred_rgb = cv2.applyColorMap(
            (depth_pred * 255).astype(np.uint8), cv2.COLORMAP_PLASMA
        )

        frame_vis = np.concatenate(
            [rgb_frame, disparity_frame_rgb, depth_pred_rgb], 0
        )
        frame_vis = cv2.resize(frame_vis, (0, 0), fx=scale, fy=scale)
        frame_vis = cv2.cvtColor(frame_vis, cv2.COLOR_BGR2RGB)

        if plot3D:
            points = frame["points"]
            points_colors = frame["points_colors"]

            point_cloud_array.put(generate_3D_frame(points, points_colors))

        if plot2D:
            plt.imshow(frame_vis)
            plt.pause(0.01)
            plt.show()

            # Check if key is pressed
            if plt.waitforbuttonpress(0.001):
                print("Key pressed, exiting...")
                exit()
