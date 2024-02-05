import json
import os
import random
import traceback
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import random_split
from tqdm import tqdm

from ..model import TrajDPT_versions, model_types
from ..model.loader import load_model, load_transforms

from ..utils import evaluate, get_batch, visualize_model_pred
from ..utils.trajectory import relative_to_absolute

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

@torch.no_grad()
def demo(
    trajectory_mode,
    load,
    TrajNet,
    TrajDPT_version,
    device,
    model_type,
    checkpoint_dir,
    dataset_name,
    project_name,
    amp,
):
    dir_checkpoint = os.path.join(checkpoint_dir, project_name)
    device_cpu = torch.device("cpu")

    # Clear memory
    torch.cuda.empty_cache()

    # REPRODUCIBILITY
    # random.seed(0)
    # np.random.seed(0)
    # torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True, warn_only=True)

    max_warp_y = 0.0
    random_flip = False

    wandb_run = True
    if wandb.run is None:
        wandb_run = False

    wandb_run_id = "dummy_run"
    if wandb_run:
        wandb_run_id = wandb.run.id
    print("wandb_run_id", wandb_run_id)

    intrinsic_matrix = np.array(
        [
            [421.733500576242, 0, 160.266326752219],
            [0, 422.052900217240, 116.910665297498],
            [0, 0, 1],
        ]
    )
    DistCoef = np.array(
        [
            -0.132676312942780, 0.0515532356737028, 0, 0, 0
        ]
    )

    # 1. Create dataset
    transform, net_w, net_h = load_transforms(
        model_type=model_type,
    )
    dataset = []
    if "lwir_raw" in dataset_name:
        from ..datasets.thermal_voyager_dataset import (
            ThermalVoyagerCarStateDataset_lwir_raw,
            get_tvd_dataset,
        )

        dataset = get_tvd_dataset(
            tvd_base="/home/shared/Thermal_Voyager",
            car_dataset_base="/home/shared/car_dataset/car_dataset",
            tvd_class=ThermalVoyagerCarStateDataset_lwir_raw,
            transform=transform,
        )
    elif "lwir_norm" in dataset_name:
        from ..datasets.thermal_voyager_dataset import (
            ThermalVoyagerCarStateDataset_lwir_norm,
            get_tvd_dataset,
        )

        dataset = get_tvd_dataset(
            tvd_base="/home/shared/Thermal_Voyager",
            car_dataset_base="/home/shared/car_dataset/car_dataset",
            tvd_class=ThermalVoyagerCarStateDataset_lwir_norm,
            transform=transform,
        )

    elif "rgb" in dataset_name:
        from TrajNet.datasets.autopilot_iterator.autopilot_carstate_iterator import (
            get_carstate_dataset,
        )

        dataset = get_carstate_dataset(
            # dataset_base="/home/shared/car_dataset/car_dataset/",
            dataset_base=os.path.expanduser("~/Datasets/car_dataset/"),
            frame_skip=0,
            MAX_WARP_Y=max_warp_y,
            random_flip=random_flip,
            chunk_size=1,
            transforms=transform,
        )

        intrinsic_matrix = np.array(
            [
                [525.5030, 0, 333.4724],
                [0, 531.1660, 297.5747],
                [0, 0, 1.0],
            ]
        )
        DistCoef = np.array(
            [
                0.0177,
                3.8938e-04,  # Tangential Distortion
                -0.1533,
                0.4539,
                -0.6398,  # Radial Distortion
            ]
        )

    assert len(dataset) > 0, "No dataset selected"

    # dataset = ConcatDataset(dataset)

    
    print("len(dataset)", len(dataset))
    assert len(dataset) > 0, "Dataset is empty"

    num_trajectory_templates = 10
    trajectory_templates_npy = f"./trajectory_templates/proposed_trajectory_templates_{num_trajectory_templates}.npy"
    trajectory_templates_kmeans_pkl = f"./trajectory_templates/kmeans_{num_trajectory_templates}.pkl"

    assert os.path.exists(trajectory_templates_npy), f"Trajectory templates file {trajectory_templates_npy} does not exist"
    assert os.path.exists(trajectory_templates_kmeans_pkl), f"Trajectory templates file {trajectory_templates_kmeans_pkl} does not exist"

    # Load net
    model_kwargs = dict(
        trajectory_mode=trajectory_mode,
        trajectory_templates=trajectory_templates_npy,
        num_trajectory_templates=num_trajectory_templates,
    )
    if TrajDPT_version == 1:
        pass

    net = load_model(
        arch=TrajNet,
        model_kwargs=model_kwargs,
        device=device_cpu,
        model_path=load,
    )
    
    net = net.to(device=device)
    # net = torch.compile(net) # causes issues

    # Clear memory
    torch.cuda.empty_cache()

    print("net", type(net))

    print("net all params")
    mem_params = sum(
        [param.nelement() * param.element_size() for param in net.parameters()]
    )
    mem_bufs = sum(
        [buf.nelement() * buf.element_size() for buf in net.buffers()]
    )
    mem = mem_params + mem_bufs  # in bytes
    print("mem", mem / 1024.0 / 1024.0, " MB")

    print("net trainable params")
    mem_params = sum(
        [
            param.nelement() * param.element_size()
            for param in net.parameters()
            if param.requires_grad
        ]
    )
    mem_bufs = sum(
        [
            buf.nelement() * buf.element_size()
            for buf in net.buffers()
            if buf.requires_grad
        ]
    )
    mem = mem_params + mem_bufs  # in bytes
    print("mem", mem / 1024.0 / 1024.0, " MB")

    if wandb_run:
        print(
            f"""Starting training:
            epochs: {wandb.config.epochs}
            batch_size: {wandb.config.batch_size}
            learning_rate: {wandb.config.learning_rate}
            val_percent: {wandb.config.val_percent}
            save_checkpoint: {wandb.config.save_checkpoint}
            amp: {wandb.config.amp}
        """
        )


    if trajectory_mode == 'regress':
        MSELoss_criterion = torch.nn.MSELoss()
        def criterion(masks_pred, true_masks):
            return MSELoss_criterion(masks_pred, true_masks)
    elif trajectory_mode == 'templates':

        # Load kmeans object from trajectory_templates/kmeans.pkl
        with open(trajectory_templates_kmeans_pkl, 'rb') as f:
            kmeans = pickle.load(f)

        CELoss_criterion = torch.nn.CrossEntropyLoss()
        trajectory_templates = net.trajectory_templates # (500, 250, 2)
        trajectory_templates = torch.tensor(
            trajectory_templates
        ).to(
            device=device,
            dtype=torch.float32
        )

        def kmeans_pred(traj_gt):
            """
                traj_gt:    (B, 250, 2) # 250 points (x,y)

                Returns:
                expected_ohe:      (B,)
            """
            # Compute gt_traj_absolute
            traj_gt = traj_gt.cpu().detach().numpy()
            traj_gt = traj_gt.astype('double')
            expected_ohe = kmeans.predict(
                traj_gt.reshape((-1, 500))
            )
            return torch.tensor(
                expected_ohe
            ).to(device=device, dtype=torch.long)


        def criterion(pred_ohe, traj_relative_gt):
            """
                trajectory_templates:   (500, 250, 2) # 500 templates, each of 250 points (x, y)
                traj_gt:                (B, 250, 2) # 250 points (x,y)
                
                trajectory_ohe:         (B, 500) # 500 classes
                expected_ohe:           (B, 500) # 500 classes
            """
            B = pred_ohe.shape[0]

            traj_relative_gt = traj_relative_gt.reshape((B, 250, 2)) # Reshaping to match the templates
            traj_gt = relative_to_absolute(traj_relative_gt) # (B, 250, 2)

            expected_ohe = kmeans_pred(traj_gt) # (B, 250, 2) # 250 points (x,y)
            
            return CELoss_criterion(pred_ohe, expected_ohe)


    batch_size = 1
    val_index = 0

    with tqdm(
        total=len(dataset), unit="img"
    ) as pbar:
        for batch_index in range(batch_size, len(dataset), batch_size):
            try:
                torch.cuda.empty_cache()
                batch = get_batch(dataset, batch_index, batch_size, N=4)

                x, frame_merged, frame_center, y = batch
                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)

                # for net_patch in PatchWiseInplace(
                #     net, patchwise_percentage
                # ):
                with torch.cuda.amp.autocast(enabled=amp):
                    y_pred = net(x) # (B, 500)
                    loss = criterion(y_pred, y)
                
                traj_pred = y_pred[val_index].cpu().detach().reshape((250, 2)).numpy()
                traj_gt = y[val_index].cpu().detach().reshape((250, 2)).numpy()

                frame_img_merged = frame_merged[val_index].cpu().detach().numpy()
                
                frame_img = visualize_model_pred(
                    traj_pred,
                    traj_gt,
                    frame_img_merged,
                    intrinsic_matrix,
                    DistCoef,
                )

                cv2.imwrite(f"./outputs/frame.png", frame_img)


                pbar.update(1)
            except Exception as ex:
                print(ex)
                traceback.print_exc()
                raise ex


def main(args):
    with open(args.sweep_json, "r") as sweep_json_file:
        sweep_config = json.load(sweep_json_file)

    trajectory_mode_vals = sweep_config["parameters"]["trajectory_mode"]["values"]

    assert len(trajectory_mode_vals) == 1, (
        "trajectory_mode can be either 'regress' or 'templates', only one per sweeep"
    )

    trajectory_mode = trajectory_mode_vals[0]

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    project_name = "{TEST}TrajDPT_V{version}_{model_type}_{dataset}_{trajectory_mode}".format(
        # TEST="TEST_",
        TEST="",
        version=str(args.version),
        model_type=args.model_type,
        dataset=args.dataset,
        trajectory_mode=trajectory_mode,
    )

    sweep_config["parameters"]["project_name"] = {"values": [project_name]}

    load = sweep_config["parameters"]["load"]
    TrajNet = TrajDPT_versions[args.version]
    version = args.version
    device = args.device
    model_type = args.model_type
    checkpoint_dir = args.checkpoint_dir
    dataset = args.dataset
    amp = False

    demo(
        trajectory_mode=trajectory_mode,
        load=load,
        TrajNet=TrajNet,
        TrajDPT_version=version,
        device=device,
        model_type=model_type,
        checkpoint_dir=checkpoint_dir,
        dataset_name=dataset,
        project_name=project_name,
        amp=amp,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TrajNet")
    parser.add_argument(
        "-v",
        "--version",
        choices=[1, 2, 3],
        required=True,
        type=int,
        help="TrajNet version",
    )

    parser.add_argument(
        "-dt",
        "--dataset",
        choices=["lwir_raw", "lwir_norm", "rgb"],
        required=True,
        help="Dataset to train using",
    )

    parser.add_argument(
        "-t",
        "--model_type",
        choices=model_types,
        required=True,
        help="Model architecture to use",
    )

    parser.add_argument(
        "-d",
        "--device",
        # default="cuda:0" if torch.cuda.is_available() else "cpu",
        default="cpu",
        help="Device to use for training",
    )

    parser.add_argument(
        "-c",
        "--checkpoint_dir",
        default=os.path.join(os.getcwd(), "checkpoints"),
        help="Directory to save checkpoints in",
    )

    parser.add_argument(
        "-b",
        "--base_path",
        default=os.path.expanduser("~/Datasets/Depth_Dataset_Bengaluru"),
        help="Base path to dataset",
    )

    parser.add_argument(
        "--sweep_json", required=True, help="Path to checkpoint to sweep json"
    )

    args = parser.parse_args()

    main(args=args)
