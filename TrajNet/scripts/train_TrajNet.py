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

from ..model import TrajDPT_versions, model_types
from ..model.loader import load_model, load_transforms

from ..utils import evaluate, get_batch
from ..utils.trajectory import relative_to_absolute

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train_net_wandb():
    experiment = wandb.init(resume="allow", anonymous="must")

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    learning_rate = wandb.config.learning_rate
    val_percent = wandb.config.val_percent
    save_checkpoint = wandb.config.save_checkpoint
    amp = wandb.config.amp
    weight_decay = wandb.config.weight_decay
    trajectory_mode = wandb.config.trajectory_mode
    patchwise_percentage = wandb.config.patchwise_percentage
    dataset_percentage = wandb.config.dataset_percentage
    random_flip = wandb.config.random_flip
    max_warp_y = wandb.config.max_warp_y
    load = wandb.config.load

    version = wandb.config.version
    device = wandb.config.device
    model_type = wandb.config.model_type
    device = torch.device(device)
    checkpoint_dir = wandb.config.checkpoint_dir
    project_name = wandb.config.project_name
    base_path = wandb.config.base_path
    dataset = wandb.config.dataset

    TrajNet = TrajDPT_versions[version]

    try:
        train_net(
            experiment=experiment,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            val_percent=val_percent,
            save_checkpoint=save_checkpoint,
            amp=amp,
            weight_decay=weight_decay,
            trajectory_mode=trajectory_mode,
            patchwise_percentage=patchwise_percentage,
            dataset_percentage=dataset_percentage,
            random_flip=random_flip,
            load=load,
            max_warp_y=max_warp_y,
            TrajNet=TrajNet,
            TrajDPT_version=version,
            device=device,
            model_type=model_type,
            checkpoint_dir=checkpoint_dir,
            base_path=base_path,
            dataset_name=dataset,
            project_name=project_name,
        )
    except Exception as ex:
        print(ex)
        traceback.print_exc()
        raise ex


def train_net(
    experiment,
    epochs,
    batch_size,
    learning_rate,
    val_percent,
    save_checkpoint,
    amp,
    weight_decay,
    trajectory_mode,
    patchwise_percentage,
    dataset_percentage,
    random_flip,
    load,
    max_warp_y,
    TrajNet,
    TrajDPT_version,
    device,
    model_type,
    checkpoint_dir,
    base_path,
    dataset_name,
    project_name,
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

    wandb_run = True
    if wandb.run is None:
        wandb_run = False

    wandb_run_id = "dummy_run"
    if wandb_run:
        wandb_run_id = wandb.run.id
    print("wandb_run_id", wandb_run_id)

    # (Initialize logging)
    if wandb_run:
        wandb.config.update(
            dict(
                amp=amp,
                epochs=epochs,
                batch_size=batch_size,
                val_percent=val_percent,
                weight_decay=weight_decay,
                learning_rate=learning_rate,
                save_checkpoint=save_checkpoint,
            )
        )

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
            dataset_base="/home/shared/car_dataset/car_dataset/",
            # dataset_base=os.path.expanduser("~/Datasets/car_dataset/"),
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
    # import torchvision
    # net = torchvision.models.resnet.ResNet(
    #     block = torchvision.models.resnet.BasicBlock, # : Type[Union[BasicBlock, Bottleneck]],
    #     layers = [3, 4, 23, 3], # resnet101
    #     # layers = [3, 4, 6, 3], # resnet50
    #     num_classes = 250 * 2, # : int = 1000,
    # )

    
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

    # 4. Set up the optimizer, the loss, the learning rate scheduler
    # and the loss scaling for AMP
    optimizer = optim.Adam(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=weight_decay,
        amsgrad=False,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2
    )  # goal: minimize the loss
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0

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
        # cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # def criterion(pred_ohe, traj_relative_gt):
        #     """
        #         trajectory_templates:   (500, 250, 2) # 500 templates, each of 250 points (x, y)
        #         traj_gt:                (B, 250, 2) # 250 points (x,y)
                
        #         trajectory_ohe:         (B, 500) # 500 classes
        #         expected_ohe:           (B, 500) # 500 classes
        #     """
        #     B = pred_ohe.shape[0]

        #     traj_relative_gt = traj_relative_gt.reshape((B, 250, 2)) # Reshaping to match the templates
        #     traj_gt = relative_to_absolute(traj_relative_gt) # (B, 250, 2)

        #     # Expand dims to prepare for broadcasting
        #     expanded_traj_gt = traj_gt.unsqueeze(1) # Shape (B, 1, 250, 2)
        #     expanded_templates = trajectory_templates.unsqueeze(0) # Shape (1, 500, 250, 2)

        #     # Compute cosine similarity between all the traj_gt and the trajectory_templates
        #     similarity = cos_sim(expanded_traj_gt.view(B, 500, -1), expanded_templates.view(1, 500, -1))

        #     # Select max similarity
        #     expected_ohe = torch.argmax(similarity, dim=1) # (B,)

        #     return CELoss_criterion(pred_ohe, expected_ohe)

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


    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(
            total=len(train_set), desc=f"Epoch {epoch}/{epochs}", unit="img"
        ) as pbar:
            for batch_index in range(batch_size, len(train_set), batch_size):
                try:
                    torch.cuda.empty_cache()
                    batch = get_batch(train_set, batch_index, batch_size, N=4)

                    x, frame_merged, frame_center, y = batch
                    x = x.to(device=device, dtype=torch.float32)
                    y = y.to(device=device, dtype=torch.float32)

                    # for net_patch in PatchWiseInplace(
                    #     net, patchwise_percentage
                    # ):
                    with torch.cuda.amp.autocast(enabled=amp):
                        y_pred = net(x) # (B, 500)
                        loss = criterion(y_pred, y)

                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()

                    pbar.update(batch_size)
                    epoch_loss += loss.item()
                    experiment.log(
                        {
                            "train_loss": loss.item(),
                            "step": global_step,
                            "epoch": epoch,
                        }
                    )
                    pbar.set_postfix(**{"loss (batch)": loss.item()})

                    # Evaluation round
                    division_step = n_train // (3 * batch_size)
                    # if division_step >= 0:
                    if global_step % division_step == 0 and wandb_run:
                        evaluate(
                            net,
                            val_set,
                            trajectory_mode,
                            device,
                            amp,
                            global_step,
                            epoch,
                            experiment,
                            intrinsic_matrix,
                            DistCoef,
                            lr=optimizer.param_groups[0]["lr"],
                        )
                        scheduler.step(loss)
                    global_step += 1
                except Exception as ex:
                    print(ex)
                    traceback.print_exc()
                    raise ex

            if save_checkpoint:
                dir_checkpoint_run = os.path.join(dir_checkpoint, wandb_run_id)
                Path(dir_checkpoint_run).mkdir(parents=True, exist_ok=True)
                torch.save(
                    net.state_dict(),
                    str(
                        os.path.join(
                            dir_checkpoint_run,
                            "checkpoint_epoch_{}.pth".format(epoch),
                        )
                    ),
                )
                print(f"Checkpoint {epoch} saved!")


def main(args):
    with open(args.sweep_json, "r") as sweep_json_file:
        sweep_config = json.load(sweep_json_file)

    sweep_config["parameters"]["device"] = {"values": [args.device]}
    sweep_config["parameters"]["version"] = {"values": [args.version]}
    sweep_config["parameters"]["model_type"] = {"values": [args.model_type]}
    sweep_config["parameters"]["checkpoint_dir"] = {
        "values": [args.checkpoint_dir]
    }
    sweep_config["parameters"]["dataset"] = {"values": [args.dataset]}
    sweep_config["parameters"]["base_path"] = {"values": [args.base_path]}
    sweep_config["parameters"]["base_path"] = {"values": [args.base_path]}

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

    sweep_id = wandb.sweep(
        sweep_config, project=project_name, entity="adityang"
    )
    print("sweep_id", sweep_id)
    wandb.agent(sweep_id, function=train_net_wandb, count=2)


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
