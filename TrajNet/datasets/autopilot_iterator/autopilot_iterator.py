import json
import os
import pickle
import traceback

import cv2
import numpy as np
import pwlf
import tqdm
from torch.utils.data import Dataset


class AutopilotIter(Dataset):
    def __init__(
        self,
        autopilot_base_path="/home/shared/car_dataset/2023-02-08_15:42:33.822505/0:5/",
    ) -> None:
        self.autopilot_base_path = autopilot_base_path
        self.transforms_json = os.path.join(
            self.autopilot_base_path, "transforms.json"
        )
        assert os.path.isfile(
            self.transforms_json
        ), "Transforms missing, run colmap.py"
        with open(self.transforms_json) as transforms_file:
            self.transforms = json.load(transforms_file)

        fl_x = self.transforms["fl_x"]
        fl_y = self.transforms["fl_y"]
        cx = self.transforms["cx"]
        cy = self.transforms["cy"]
        self.camera_intrinsics = np.array(
            [
                [fl_x, 0.0, cx, 0.0],
                [0.0, fl_y, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.trajectory = []
        for frame_index in range(len(self.transforms["frames"])):
            frame = self.transforms["frames"][frame_index]
            file_path = frame["file_path"].split("/")[-1]
            frame_img_path = os.path.join(
                self.autopilot_base_path, "rgb_2.mp4", file_path
            )
            assert os.path.isfile(frame_img_path), (
                "Frame missing" + frame["file_path"]
            )

            self.trajectory.append(np.array(frame["transform_matrix"]))

        self.index = 0
        self.frame_lookahead = min(50, len(self))

        # self.frames_sorted = self.trajectory_by_distance()
        # self.frame_lookahead = len(self)

    def __len__(self):
        # TODO: Implement
        return len(self.transforms["frames"])

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        data = self[self.index]
        self.index += 1
        return data

    def trajectory_by_distance(self, frame, cutoff_dist=float("inf")):
        point = np.array(frame["transform_matrix"])
        trajectory = self.transforms["frames"].copy()

        def dist(frame_2):
            # TODO: return inf for points behind camera
            point_2 = np.array(frame_2["transform_matrix"])
            return np.linalg.norm(point[:3, 3] - point_2[:3, 3])

        trajectory.sort(key=dist)
        # distances = [dist(x) for x in trajectory]
        # print('min, max', min(distances), max(distances)) # upto 11 meters
        trajectory = [x for x in trajectory if dist(x) < cutoff_dist]
        return trajectory

    def __getitem__(self, frame_index):
        frame = self.transforms["frames"][frame_index]
        file_path = frame["file_path"].split("/")[-1]
        frame_img_path = os.path.join(
            self.autopilot_base_path, "rgb_2.mp4", file_path
        )
        frame_img_left_path = frame_img_path.replace("rgb_2.mp4", "rgb_1.mp4")
        frame_img_right_path = frame_img_path.replace("rgb_2.mp4", "rgb_3.mp4")
        frame_image = cv2.imread(frame_img_path)
        frame_image_left = cv2.imread(frame_img_left_path)
        frame_image_right = cv2.imread(frame_img_right_path)

        transform_matrix = np.array(frame["transform_matrix"])
        trajectory = self.trajectory_by_distance(frame, cutoff_dist=3.0)[
            : self.frame_lookahead
        ]

        transform_matricies = []
        for i in range(len(trajectory)):
            frame_i = trajectory[i]
            transform_matricies.append(np.array(frame_i["transform_matrix"]))

        return {
            "frame_image": frame_image,
            "frame_image_left": frame_image_left,
            "frame_image_right": frame_image_right,
            "transform_matrix": transform_matrix,
            "transform_matricies": transform_matricies,
        }


class AutopilotIterTraj(Dataset):
    def __init__(
        self,
        autopilot_base_path="/home/shared/car_dataset/2023-02-08_15:42:33.822505/0:5/",
        n_points=10,
        s=4,
        xfit_range=(0.0, 0.6),
        HEIGHT_CORRECTION=0.5,
        load_cache=True,
    ) -> None:
        assert len(xfit_range) == 2, "xfit_range must be a tuple of size=2"
        assert type(s) == int, "s must be an int"
        assert type(n_points) == int, "n_points must be an int"

        self.dataset = AutopilotIter(autopilot_base_path)
        self.load_cache = load_cache
        self.cache_path = os.path.join(
            autopilot_base_path,
            "cache_{HEIGHT_CORRECTION}.pkl".format(
                HEIGHT_CORRECTION=HEIGHT_CORRECTION
            ),
        )

        self.HEIGHT_CORRECTION = HEIGHT_CORRECTION
        self.data = []

        if os.path.isfile(self.cache_path) and self.load_cache:
            with open(self.cache_path, "rb") as cache_file:
                self.data = pickle.load(cache_file)
        else:
            for frame_index in tqdm.tqdm(range(len(self.dataset))):
                try:
                    frame = self.dataset[frame_index]
                    final_frame = frame["frame_image"].copy()
                    WIDTH, HEIGHT, _ = frame["frame_image"].shape
                    Tab = frame["transform_matrix"]
                    traj_2d = []
                    imgx = []
                    imgy = []
                    for frame_lookahead in range(
                        len(frame["transform_matricies"])
                    ):
                        Tac = frame["transform_matricies"][frame_lookahead]
                        Tbc = np.linalg.inv(Tac) @ Tab

                        p3d = Tbc[:, 3].reshape((4, 1))
                        # p3d[2,0] = p3d[2,0] - self.HEIGHT_CORRECTION
                        p3d[1, 0] = p3d[1, 0] - 0.2
                        # p3d[0,0] = p3d[0,0] - 0.2
                        # p3d[0,0] = -p3d[0,0]
                        # p3d[0,0] = p3d[0,0] + 0.2
                        # p3d[2,0] = p3d[2,0]
                        # print('z', p3d[2,0])
                        if (
                            p3d[2, 0]
                            < 0  # Ensure the point is infront of the camera
                        ):
                            p2d = self.dataset.camera_intrinsics @ p3d
                            if p2d[2][0] != 0.0:
                                px, py = round(p2d[0][0] / p2d[2][0]), round(
                                    p2d[1][0] / p2d[2][0]
                                )
                                if 0 <= px < WIDTH and 0 <= py < HEIGHT:
                                    # if 0 <= px < HEIGHT and 0 <= py < WIDTH:
                                    # print((px, py))
                                    imgx += [px]
                                    imgy += [py]
                                    # final_frame = cv2.circle(final_frame, (px, py), 5, (0,255,0), -1)
                                    # final_frame = cv2.circle(final_frame, (WIDTH-1, HEIGHT-1), 5, (0,255,0), -1)
                                    # final_frame = cv2.circle(final_frame, (HEIGHT-1, WIDTH-1), 5, (0,255,0), -1)
                                    traj_2d.append(
                                        [
                                            p3d[0, 0],
                                            p3d[1, 0],
                                        ]
                                    )

                    traj_2d = sorted(
                        traj_2d, key=lambda traj_point: traj_point[1]
                    )  # sort by x axis

                    traj_2d = np.array(traj_2d).reshape(-1, 2)

                    y, x = traj_2d[:, 0], traj_2d[:, 1]

                    # Scipy Mentod
                    knot_numbers = 5
                    x_new = np.linspace(0, 1, knot_numbers + 2)[1:-1]
                    xfit = np.linspace(xfit_range[0], xfit_range[1], n_points)

                    xfit_rel = []
                    yfit_rel = []
                    yfit = []
                    if len(x) > 0:
                        my_pwlf = pwlf.PiecewiseLinFit(x, y)

                        # fit the data for four line segments
                        # print(frame_index, len(x), n_points, min(len(x), n_points))
                        # res = my_pwlf.fit(min(len(x), n_points))
                        # res = my_pwlf.fit(5)
                        # res = my_pwlf.fitfast(5)
                        # res = my_pwlf.fitfast(min(len(x), n_points))

                        res = my_pwlf.fit(len(x))
                        # res = my_pwlf.fitfast(len(x))

                        yfit = my_pwlf.predict(xfit)

                        xfit_rel = [xfit[0]]
                        yfit_rel = [yfit[0]]
                        for fit_i in range(1, len(xfit)):
                            xfit_rel.append(xfit[fit_i] - xfit[fit_i - 1])
                            yfit_rel.append(yfit[fit_i] - yfit[fit_i - 1])

                        self.data.append(
                            {
                                "frame_image": frame["frame_image"],
                                "frame_image_left": frame["frame_image_left"],
                                "frame_image_right": frame[
                                    "frame_image_right"
                                ],
                                "transform_matrix": frame["transform_matrix"],
                                "transform_matricies": frame[
                                    "transform_matricies"
                                ],
                                "xfit": xfit,
                                "yfit": yfit,
                                "x": x,
                                "y": y,
                                "xfit_rel": xfit_rel,
                                "yfit_rel": yfit_rel,
                                "imgx": imgx,
                                "imgy": imgy,
                            }
                        )
                except Exception as ex:
                    print(ex)
                    traceback.print_exc()
                    print(len(frame["transform_matrix"]))
                    exit()
        with open(self.cache_path, "wb") as cache_file:
            pickle.dump(self.data, cache_file)

    def __len__(self):
        # TODO: Implement
        return len(self.data)

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
        return self.data[frame_index]


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


def main():

    # autopilot_datasets = get_all_autopilot_traj_datasets("/home/shared/car_dataset/") # 11 seconds
    autopilot_datasets = get_all_autopilot_traj_datasets(
        "/home/aditya/Datasets/car_dataset"
    )
    # autopilot_datasets = get_all_autopilot_traj_datasets("/home/shared/car_dataset/2023-02-08_15:42:33.822505") # 30 seconds

    # autopilot_datasets = get_all_autopilot_traj_datasets("/home/aditya/Datasets/car_dataset/2023-02-08_15:42:33.822505")

    # autopilot_datasets = [
    #     AutopilotIterTraj("/home/shared/car_dataset/2023-02-08_15:42:33.822505/0:5"),
    #     AutopilotIterTraj("/home/shared/car_dataset/2023-02-08_15:42:33.822505/5:10"),
    #     AutopilotIterTraj("/home/shared/car_dataset/2023-02-08_15:42:33.822505/10:15"),
    #     AutopilotIterTraj("/home/shared/car_dataset/2023-02-08_15:42:33.822505/15:25"),
    # ]

    # autopilot_datasets = [
    #     AutopilotIterTraj("/home/shared/car_dataset/2023-02-08_15:42:33.822505/0:5", load_cache=False),
    # ]

    # autopilot_datasets = get_all_autopilot_datasets("/home/shared/car_dataset/")

    # autopilot_datasets = [
    #     AutopilotIterTraj("/home/aditya/Datasets/car_dataset/2023-02-08_15:42:33.822505/0:5/", load_cache=False),
    # ]
    for dataset in autopilot_datasets:
        # print(dataset.dataset.autopilot_base_path, len(dataset))
        for frame_index in range(len(dataset)):
            frame = dataset[frame_index]

            final_frame = frame["frame_image"]

            xfit = frame["xfit"]
            yfit = frame["yfit"]

            xfit_rel = frame["xfit_rel"]
            yfit_rel = frame["yfit_rel"]

            x_rec = [xfit_rel[0]]
            y_rec = [yfit_rel[0]]

            for rec_i in range(1, len(xfit_rel)):
                x_rec.append(xfit_rel[rec_i] + x_rec[rec_i - 1])
                y_rec.append(yfit_rel[rec_i] + y_rec[rec_i - 1])

            assert len(x_rec) == len(xfit)
            for rec_i in range(0, len(x_rec)):
                # print(x_rec[rec_i] - xfit[rec_i], y_rec[rec_i] - yfit[rec_i])
                assert np.isclose(x_rec[rec_i], xfit[rec_i])
                assert np.isclose(y_rec[rec_i], yfit[rec_i])

            for p2d_i in range(len(frame["imgx"])):
                imgx = frame["imgx"][p2d_i]
                imgy = frame["imgy"][p2d_i]
                final_frame = cv2.circle(
                    final_frame, (imgx, imgy), 5, (0, 255, 0), -1
                )

            cv2.imshow("frame", final_frame)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    return

    hi = AutopilotIter()

    for frame in hi:
        final_frame = frame["frame_image"]
        Tab = frame["transform_matrix"]

        heights = []

        for frame_lookahead in range(len(frame["transform_matricies"])):
            Tac = frame["transform_matricies"][frame_lookahead]
            Tbc = np.linalg.inv(Tac) @ Tab

            p3d = Tbc[:, 3].reshape((4, 1))
            p3d[2, 0] = p3d[2, 0] - 0.5
            # p3d = (Tbc @ Tac[:,3]).reshape((4,1))
            if p3d[2, 0] > 0:

                heights.append(p3d[2, 0])

                p2d = hi.camera_intrinsics @ p3d
                if p2d[2][0] != 0.0:
                    px, py = round(p2d[0][0] / p2d[2][0]), round(
                        p2d[1][0] / p2d[2][0]
                    )
                    final_frame = cv2.circle(
                        final_frame, (px, py), 5, (0, 255, 0), -1
                    )
                    # print('px,py', px,py)

        print(
            "heights: min, max, avg",
            min(heights),
            max(heights),
            sum(heights) / float(len(heights)),
        )
        cv2.imshow("frame", final_frame)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
