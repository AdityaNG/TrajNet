import numpy as np
import torch
from dataset_helper.dataset_iterators import BengaluruDepthDatasetIterator


class BDD_RGB_Boosted(BengaluruDepthDatasetIterator):
    """
    Bengaluru Depth Dataset
        RGB data
        Boosted depth
    """

    def __init__(
        self,
        dataset_path="~/Datasets/Depth_Dataset_Bengaluru/1653972957447",
        settings_doc="~/Datasets/Depth_Dataset_Bengaluru/calibration/pocoX3/calib.yaml",
        transform=None,
    ):
        super().__init__(dataset_path=dataset_path, settings_doc=settings_doc)
        self.img_transform = transform

    def __getitem__(self, frame_index):
        frame = super().__getitem__(frame_index)

        rgb_frame = frame["rgb_frame"]
        disparity_frame = frame["disparity_frame"]
        # csv_frame = frame['csv_frame']

        x = self.img_transform(rgb_frame)
        x_raw = torch.tensor(rgb_frame).unsqueeze(
            0
        )  # Image used for visualization
        y = torch.tensor(disparity_frame).unsqueeze(0)
        mask = torch.ones_like(y, dtype=torch.bool)

        return [x, x_raw, mask, y]


##################################################################################


def normalize(mat):
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
