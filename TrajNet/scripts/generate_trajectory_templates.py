import json
import os
import random
import traceback
import pickle
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ..model import TrajDPT_versions, model_types
from ..model.loader import load_model, load_transforms

from ..utils import evaluate, get_batch
from ..utils.trajectory import relative_to_absolute

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


@torch.no_grad()
def generate_trajectory_templates(
    val_percent,
    dataset_percentage,
    dataset_name,
):
    device = torch.device("cpu")

    # Clear memory
    torch.cuda.empty_cache()

    # REPRODUCIBILITY
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    
    # 1. Create dataset
    identity_transform = lambda x: x
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
            transforms=identity_transform,
        )
    elif "lwir_norm" in dataset_name:
        from ..datasets.thermal_voyager_dataset import (
            ThermalVoyagerCarStateDataset_lwir_raw,
            get_tvd_dataset,
        )

        dataset = get_tvd_dataset(
            tvd_base="/home/shared/Thermal_Voyager",
            car_dataset_base="/home/shared/car_dataset/car_dataset",
            tvd_class=ThermalVoyagerCarStateDataset_lwir_raw,
            transforms=identity_transform,
        )

    elif "rgb" in dataset_name:
        from TrajNet.datasets.autopilot_iterator.autopilot_carstate_iterator import (
            get_carstate_dataset,
        )

        dataset = get_carstate_dataset(
            dataset_base="/home/shared/car_dataset/car_dataset/",
            # dataset_base=os.path.expanduser("~/Datasets/car_dataset/"),
            frame_skip=0,
            MAX_WARP_Y=0.1,
            random_flip=True,
            chunk_size=1,
            transforms=identity_transform,
        )

    assert len(dataset) > 0, "No dataset selected"

    # dataset = ConcatDataset(dataset)

    total_size = len(dataset)
    total_use = int(round(total_size * dataset_percentage))
    total_discard = total_size - total_use
    dataset, _ = random_split(
        dataset,
        [total_use, total_discard],
        generator=torch.Generator().manual_seed(0),
    )
    print("len(dataset)", len(dataset))
    assert len(dataset) > 0, "Dataset is empty"

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    assert n_val > 0, "Validation count is 0"
    assert n_train > 0, "Train count is 0"

    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    assert len(val_set) > 0, "Validation set is 0"
    assert len(train_set) > 0, "Train set is 0"

    # Clear memory
    torch.cuda.empty_cache()

    batch_size = 1
    K = 15
    proposed_trajectory_templates = []

    # Collecting trajectories for clustering
    trajectories = []
    with tqdm(
        total=len(train_set), desc=f"Generating trajectory templates", unit="img"
    ) as pbar:
        for batch_index in range(batch_size, len(train_set), batch_size):
            batch = get_batch(train_set, batch_index, batch_size, N=4)
            _, _, _, trajectory_torch_relative = batch

            if len(trajectory_torch_relative.shape) == 2:
                trajectory_torch_relative = trajectory_torch_relative.unsqueeze(0)

            trajectory_torch_relative = trajectory_torch_relative.cpu().detach().reshape((-1, 250, 2))
            trajectory_torch  = relative_to_absolute(
                trajectory_torch_relative
            )
            trajectory = trajectory_torch[0].cpu().detach().reshape((250, 2)).numpy()

            trajectories.append(trajectory.flatten())  # Flattening the trajectory
            pbar.update(batch_size)

    # Performing k-means clustering
    kmeans = KMeans(n_clusters=K, random_state=0).fit(trajectories)
    proposed_trajectory_templates = [centroid.reshape((250, 2)) for centroid in kmeans.cluster_centers_]

    # Save the kmeans object as trajectory_templates/kmeans.pkl
    with open(f'trajectory_templates/kmeans_{str(K)}.pkl', 'wb') as f:
        pickle.dump(kmeans, f)

    # Plotting the trajectory templates
    plt.figure(figsize=(10, 6))
    for template in proposed_trajectory_templates:
        plt.plot(template[:, 0], template[:, 1], alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Proposed Trajectory Templates')
    plt.savefig(f'trajectory_templates/trajectory_templates_{str(K)}.png')  # Saving the plot
    plt.show()

    # Saving the templates as a NumPy file
    proposed_trajectory_templates_np = np.array(proposed_trajectory_templates, dtype=np.float32)
    print('proposed_trajectory_templates_np.shape', proposed_trajectory_templates_np.shape)
    np.save(f'trajectory_templates/proposed_trajectory_templates_{str(K)}.npy', proposed_trajectory_templates_np, allow_pickle=False)


    # Validation
    with tqdm(
        total=len(val_set), desc=f"Validating trajectory templates", unit="img"
    ) as pbar:
        for batch_index in range(batch_size, len(val_set), batch_size):
            batch = get_batch(val_set, batch_index, batch_size, N=4)
            _, _, _, trajectory_torch = batch
            trajectory = trajectory_torch[0].cpu().detach().reshape((250, 2)).numpy()

            # Evaluate the proposed_trajectory_templates
            distances = cdist([trajectory.flatten()], [template.flatten() for template in proposed_trajectory_templates])
            min_distance = np.min(distances)
            # TODO: Use min_distance for further evaluation

            pbar.update(batch_size)
    print('min_distance', min_distance)



def main(args):

    generate_trajectory_templates(
        val_percent=args.val_percent,
        dataset_percentage=args.dataset_percentage,
        dataset_name=args.dataset,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TrajNet")

    parser.add_argument(
        "-dt",
        "--dataset",
        choices=["lwir_raw", "lwir_norm", "rgb"],
        required=True,
        help="Dataset to train using",
    )

    parser.add_argument(
        "-v",
        "--val_percent",
        type=float,
        default=0.05,
        help="Dataset to train using",
    )

    parser.add_argument(
        "-p",
        "--dataset_percentage",
        type=float,
        default=1.00,
        help="Dataset to train using",
    )

    args = parser.parse_args()

    main(args=args)
