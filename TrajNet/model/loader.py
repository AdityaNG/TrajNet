from typing import Type, Union

import cv2
import torch
from torchvision.transforms import Compose

from .base_model import BaseModel
from .dpt import DPT, DPTDepthModel, DPTSegmentationModel
from .TrajNet import TrajNet
from .transforms import NormalizeImage, PrepareForNet, Resize


def load_model(
    arch: Type[Union[TrajNet, DPT, DPTDepthModel, DPTSegmentationModel]],
    model_kwargs: dict,
    device: torch.device,
    model_path: str,
    model_type: str = "dpt_large_384",
    optimize: bool = False,
) -> Type[BaseModel]:
    """Load the specified network.

    Args:
        device (device): the torch device used
        model_path (str): path to saved model
        model_type (str): the type of the model to be loaded
        optimize (bool): optimize the model to half-integer on CUDA?

    Returns:
        The loaded network
    """
    assert issubclass(
        arch, BaseModel
    ), f"arch '{arch}' not implemented, must be an instance of \
        TrajNet.model.dpt.DPT"

    if model_type == "dpt_beit_large_512":
        model = arch(
            path=model_path,
            backbone="beitl16_512",
            **model_kwargs,
        )

    elif model_type == "dpt_beit_large_384":
        model = arch(
            path=model_path,
            backbone="beitl16_384",
            **model_kwargs,
        )

    elif model_type == "dpt_beit_base_384":
        model = arch(
            path=model_path,
            backbone="beitb16_384",
            **model_kwargs,
        )

    elif model_type == "dpt_swin2_large_384":
        model = arch(
            path=model_path,
            backbone="swin2l24_384",
            **model_kwargs,
        )

    elif model_type == "dpt_swin2_base_384":
        model = arch(
            path=model_path,
            backbone="swin2b24_384",
            **model_kwargs,
        )

    elif model_type == "dpt_swin2_tiny_256":
        model = arch(
            path=model_path,
            backbone="swin2t16_256",
            **model_kwargs,
        )

    elif model_type == "dpt_swin_large_384":
        model = arch(
            path=model_path,
            backbone="swinl12_384",
            **model_kwargs,
        )

    elif model_type == "dpt_next_vit_large_384":
        model = arch(
            path=model_path,
            backbone="next_vit_large_6m",
            **model_kwargs,
        )

    # We change the notation from dpt_levit_224 (MiDaS notation) to levit_384
    # (timm notation) here, where the 224 refers to the resolution 224x224
    # used by LeViT and 384 is the first entry of the embed_dim, see
    # _cfg and model_cfgs of
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/levit.py
    # (commit id: 927f031293a30afb940fff0bee34b85d9c059b0e)
    elif model_type == "dpt_levit_224":
        model = arch(
            path=model_path,
            backbone="levit_384",
            head_features_1=64,
            head_features_2=8,
            **model_kwargs,
        )

    elif model_type == "dpt_large_384":
        model = arch(
            path=model_path,
            backbone="vitl16_384",
            **model_kwargs,
        )

    elif model_type == "dpt_hybrid_384":
        model = arch(
            path=model_path,
            backbone="vitb_rn50_384",
            **model_kwargs,
        )

    else:
        print(f"model_type '{model_type}' not implemented")
        assert False

    print(
        "Model loaded, number of parameters = {:.0f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6
        )
    )

    if optimize and (device == torch.device("cuda")):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)

    return model


def load_transforms(
    model_type: str = "dpt_large_384", height: int = 0, square: bool = False
):
    """Load the transformation for the specified network.

    Args:
        model_type (str): the type of the model to be loaded
        height (int): inference encoder image height
        square (bool): resize to a square resolution?

    Returns:
        The the transform which prepares images as input to the network and
        the dimensions of the network input
    """

    keep_aspect_ratio = not square

    if model_type == "dpt_beit_large_512":
        net_w, net_h = 512, 512
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_beit_large_384":
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_beit_base_384":
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_swin2_large_384":
        # net_w, net_h = 384, 384
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_swin2_base_384":
        # net_w, net_h = 384, 384
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_swin2_tiny_256":
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_swin_large_384":
        # net_w, net_h = 384, 384
        net_w, net_h = 256, 256
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_next_vit_large_384":
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    # We change the notation from dpt_levit_224 (MiDaS notation) to levit_384
    # (timm notation) here, where the 224 refers to the resolution 224x224
    # used by LeViT and 384 is the first entry of the embed_dim, see
    # _cfg and model_cfgs of
    # https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/levit.py
    # (commit id: 927f031293a30afb940fff0bee34b85d9c059b0e)
    elif model_type == "dpt_levit_224":
        net_w, net_h = 224, 224
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_large_384":
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    elif model_type == "dpt_hybrid_384":
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    else:
        print(f"model_type '{model_type}' not implemented")
        assert False

    if height != 0:
        net_w, net_h = height, height

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return transform, net_w, net_h
