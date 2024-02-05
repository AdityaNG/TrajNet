import os
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import yaml

from .base_model import BaseModel
from .blocks import Interpolate
from .dpt import DPT, DPTDepthModel, DPTSegmentationModel

from ..utils.trajectory import absolute_to_relative
# from .backbones.vit_3d import ViT3D


cpu_device = torch.device("cpu")

DEFAULT_CAMERA_INTRINSICS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

default_depth_models = {
    "dpt_beit_large_512": "weights/dpt_beit_large_512.pt",
    "dpt_beit_large_384": "weights/dpt_beit_large_384.pt",
    "dpt_beit_base_384": "weights/dpt_beit_base_384.pt",
    "dpt_swin2_large_384": "weights/dpt_swin2_large_384.pt",
    "dpt_swin2_base_384": "weights/dpt_swin2_base_384.pt",
    "dpt_swin2_tiny_256": "weights/dpt_swin2_tiny_256.pt",
    "dpt_swin_large_384": "weights/dpt_swin_large_384.pt",
    "dpt_next_vit_large_384": "weights/dpt_next_vit_large_384.pt",
    "dpt_levit_224": "weights/dpt_levit_224.pt",
    "dpt_large_384": "weights/dpt_large_384.pt",
    "dpt_hybrid_384": "weights/dpt_hybrid_384.pt",
}

default_seg_models = {
    "dpt_beit_large_512": None,
    "dpt_beit_large_384": None,
    "dpt_beit_base_384": None,
    "dpt_swin2_large_384": None,
    "dpt_swin2_base_384": None,
    "dpt_swin2_tiny_256": None,
    "dpt_swin_large_384": None,
    "dpt_next_vit_large_384": None,
    "dpt_levit_224": None,
    "dpt_large_384": None,
    "dpt_hybrid_384": None,
}

model_types = default_depth_models.keys()


class TrajNet(BaseModel):
    def __init__(
        self,
        model_type="dpt_swin2_tiny_256",
        backbone="swin2t16_256",
        trajectory_mode='regress',
        num_trajectory_templates=500,
        trajectory_templates="./trajectory_templates/proposed_trajectory_templates_500.npy",
        path=None,
        num_classes: int = 3,
        depth_scale=1.0,
        depth_shift=0.0,
        **kwargs
    ):
        super(TrajNet, self).__init__(**kwargs)

        assert trajectory_mode in (
            'regress',      # Have the NN predict a relative trajectory
            'templates',    # Use OHE to select a trajectory from the set of templates
        )

        ##########################
        # Load constants
        self.backbone = backbone
        self.model_type = model_type
        self.path = path
        self.scale = depth_scale
        self.shift = depth_shift
        self.num_classes = num_classes
        self.trajectory_mode = trajectory_mode

        if self.trajectory_mode == 'templates':
            self.num_trajectory_templates = num_trajectory_templates
            self.trajectory_templates = np.load(
                trajectory_templates,
                allow_pickle=False
            )
            # Validate
            assert (
                self.trajectory_templates.shape ==
                (self.num_trajectory_templates, 250, 2)
            ), f"Expected trajectory_templates.shape to be ({self.num_trajectory_templates}, 250, 2), got {self.trajectory_templates.shape}"

        features = kwargs["features"] if "features" in kwargs else 256
        self.features = features
        ##########################
        # Loading camera intrinsics
        # TODO
        ##########################

    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, 3, H, W)
        (H, W) are determined by architecture,
        look at TrajNet.loader.load_transforms for mode details
        """
        assert (
            False
        ), "Not implemented, take input batch and produce inv_depth, \
            segmentation and call \
            self.get_semantic_occupancy(inv_depth, segmentation)"

        # inv_depth = self.depth_net(depth_input)
        # segmentation = self.seg_net(seg_input)
        # return self.get_semantic_occupancy(inv_depth, segmentation)


class TrajDPT_V1(TrajNet):
    def __init__(self, **kwargs):
        super(TrajDPT_V1, self).__init__(**kwargs)

        from .loader import load_model

        ##########################
        # Loading network
        self.encoder = load_model(
            DPT,
            dict(
                head=nn.Identity(),
                non_negative=True,
            ),
            device=cpu_device,
            model_path=None,
            model_type=self.model_type,
        )

        final_features_dim = 500
        if self.trajectory_mode == 'templates':
            final_features_dim = self.num_trajectory_templates

        self.decoder = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Convolutional Layer 2
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Convolutional Layer 3
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Flatten the tensor
            nn.Flatten(),
            # Fully Connected Layer 1
            nn.Linear(512 * 16 * 16, 1024),
            nn.ReLU(),
            # Fully Connected Layer 2
            nn.Linear(1024, final_features_dim),
            # Final output reshaped to (final_features_dim,1)
            nn.Linear(final_features_dim, final_features_dim),
            nn.Unflatten(1, (final_features_dim, ))
        )

        ##########################
        ##########################
        # Load model weights
        self.load_net(self.path)

    def forward(self, x: torch.Tensor):
        device = self.get_device()

        B = x.shape[0]

        features = self.encoder(x)
        trajectory_features = self.decoder(features)

        if self.trajectory_mode == 'regress':
            return trajectory_features
        elif self.trajectory_mode == 'templates':
            trajectory_softmax = torch.softmax(
                trajectory_features,
                dim=1
            ) # (B, final_features_dim)
            return trajectory_softmax


class TrajDPT_V2(TrajNet):
    def __init__(self, **kwargs):
        super(TrajDPT_V2, self).__init__(**kwargs)

        ##########################
        # Loading network
        import torchvision

        final_features_dim = 500
        if self.trajectory_mode == 'templates':
            final_features_dim = self.num_trajectory_templates

        self.encoder = torchvision.models.resnet.ResNet(
            block = torchvision.models.resnet.BasicBlock, # : Type[Union[BasicBlock, Bottleneck]],
            layers = [3, 4, 6, 3], # resnet50
            num_classes = final_features_dim,
        )
        ##########################
        ##########################
        # Load model weights
        self.load_net(self.path)

    def forward(self, x: torch.Tensor):
        device = self.get_device()

        B = x.shape[0]

        trajectory_features = self.encoder(x)

        if self.trajectory_mode == 'regress':
            return trajectory_features
        elif self.trajectory_mode == 'templates':
            trajectory_softmax = torch.softmax(
                trajectory_features,
                dim=1
            ) # (B, 500)
            return trajectory_softmax


class TrajDPT_V3(TrajNet):
    def __init__(self, **kwargs):
        super(TrajDPT_V3, self).__init__(**kwargs)

        ##########################
        # Loading network
        import torchvision

        final_features_dim = 500
        if self.trajectory_mode == 'templates':
            final_features_dim = self.num_trajectory_templates

        self.encoder = torchvision.models.resnet.ResNet(
            block = torchvision.models.resnet.BasicBlock, # : Type[Union[BasicBlock, Bottleneck]],
            layers = [3, 4, 23, 3], # resnet101
            num_classes = final_features_dim,
        )
        ##########################
        ##########################
        # Load model weights
        self.load_net(self.path)

    def forward(self, x: torch.Tensor):
        device = self.get_device()

        B = x.shape[0]

        trajectory_features = self.encoder(x)

        if self.trajectory_mode == 'regress':
            return trajectory_features
        elif self.trajectory_mode == 'templates':
            trajectory_softmax = torch.softmax(
                trajectory_features,
                dim=1
            ) # (B, 500)
            return trajectory_softmax

TrajDPT_versions = {
    1: TrajDPT_V1,
    2: TrajDPT_V2,
    3: TrajDPT_V3,
}
