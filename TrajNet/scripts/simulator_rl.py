import numpy as np
import torch
import cv2

import json
import os
import random
import traceback
import pickle
from pathlib import Path

import torch
import wandb
from torch import optim
from torch.nn import functional as F
from torch.utils.data import random_split
from tqdm import tqdm

from ..model import TrajDPT_versions, model_types
from ..model.loader import load_model, load_transforms

from ..utils import (
    evaluate_video, compute_template_trajectory, visualize_model_pred, get_batch
)
from ..utils.trajectory import relative_to_absolute

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def warp_affine_2D_torch(image, affine_matrix, output_shape=None):
    """
    Applies 2D affine transformation to the image tensor.
    
    Args:
        image (torch.Tensor): input image tensor of shape (B, C, H, W)
        affine_matrix (torch.Tensor): affine transformation matrix of shape (B, 3, 3)
        output_shape (tuple): shape of the output image in the format (H, W). 
                               If None, it will be set to the input image shape.
                               
    Returns:
        torch.Tensor: transformed image tensor of shape (B, C, H, W)
    """
    B, C, H, W = image.shape
    
    if output_shape is None:
        output_H, output_W = H, W
    else:
        output_H, output_W = output_shape
        
    # Create normalized grid for the output image
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, output_H),
        torch.linspace(-1, 1, output_W)
    )
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0)  # Shape: (1, output_H, output_W, 2)
    grid = grid.to(image.device)
    
    # Extract the relevant sub-matrix for 2D affine transformation
    affine_matrix_2d = affine_matrix[:, :2, :3]  # Shape: (B, 2, 3)
    
    # Transform grid using the given affine matrix
    affine_grid = F.affine_grid(affine_matrix_2d, size=(B, C, output_H, output_W), align_corners=False)
    
    # Sample from the input image using the transformed grid
    warped_image = F.grid_sample(image, affine_grid, align_corners=False)
    
    return warped_image

@torch.no_grad()
def test_rl_sim(
    dataset_name='rgb'
):
    model_type = 'dpt_swin2_tiny_256'
    max_warp_y = 0.0
    random_flip = False
    device = torch.device('cuda')
    max_distance = 0.5
    max_iters = 5
    trajectory_mode = 'regress'
    version = 1
    TrajNet = TrajDPT_versions[version]
    load = "checkpoints_pretrained/TrajDPT_V1_dpt_swin2_tiny_256_rgb_regress/tki85ua1/checkpoint_epoch_15.pth"
    num_trajectory_templates = 10
    trajectory_templates_npy = f"./trajectory_templates/proposed_trajectory_templates_{num_trajectory_templates}.npy"
    trajectory_templates_kmeans_pkl = f"./trajectory_templates/kmeans_{num_trajectory_templates}.pkl"

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

    model_kwargs = dict(
        trajectory_mode=trajectory_mode,
        trajectory_templates=trajectory_templates_npy,
        num_trajectory_templates=num_trajectory_templates,
    )

    net = load_model(
        arch=TrajNet,
        model_kwargs=model_kwargs,
        device=device,
        model_path=load,
    )
    
    net = net.to(device=device)
    
    batch = get_batch(dataset, batch_index=0, batch_size=1, N=4)

    x, frame_merged, frame_center, y = batch
    x = x.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)


    loss, video_frames = rl_sim(
        net=net,            # torch.nn.Module
        frame=frame_center[0].detach().cpu().numpy(),          # np.array: (H, W, C)
        trajectory_mode=trajectory_mode,# str: 'regress' or 'templates' 
        trajectory_gt=y[0].reshape(250,2).detach().cpu().numpy(),  # np.array: (250, 2) # 250 points (x,y)
        img_transform=transform,  # Callable: used to transform the image to torch tensor for the network
        max_distance=max_distance,   # float: maximum distance to simulate to 
        max_iters=max_iters,      # int: maximum number of iterations to simulate
        K=intrinsic_matrix,              # np.array: (3, 3) Intrinsic Matrix
        D=DistCoef,              # np.array: (5, ) Distortion Coefficients
        device=device,         # torch.device: device to run the network on
    )

def rl_sim(
    net,            # torch.nn.Module
    frame,          # np.array: (H, W, C)
    trajectory_mode,# str: 'regress' or 'templates' 
    trajectory_gt,  # np.array: (250, 2) # 250 points (x,y)
    img_transform,  # Callable: used to transform the image to torch tensor for the network
    max_distance,   # float: maximum distance to simulate to 
    max_iters,      # int: maximum number of iterations to simulate
    K,              # np.array: (3, 3) Intrinsic Matrix
    D,              # np.array: (5, ) Distortion Coefficients
    device,         # torch.device: device to run the network on
):

    frame_orig = frame.copy()
    traj_gt_orig = trajectory_gt.copy()

    distance_travelled = 0.0
    iter_count = 0

    M_3D = np.eye(4)
    M_2D = np.eye(3) # or transform_3D_to_2D(M_3D, K)

    # M_2D[:,0] *= 2
    # M_2D[:,1] *= 2

    # M_3D = transform_2D_to_3D(M_2D, K)

    # print('M_2D')
    # print(M_2D)
    # print('M_3D')
    # print(M_3D)

    # exit()

    M_2D_tensor = torch.tensor(
        M_2D,
        device=device,
        dtype=torch.float32,
    ).reshape((1, 3, 3))

    M_3D_tensor = torch.tensor(
        M_3D,
        device=device,
        dtype=torch.float32,
    ).reshape((1, 4, 4))

    loss = torch.tensor(0.0, device=device, dtype=torch.float32)


    # trajectory_gt_tensor tensor(1, 250, 2)
    trajectory_gt_tensor = torch.tensor(
        trajectory_gt,
        device=device,
        dtype=torch.float32,
    ).reshape((1, 250, 2))

    # image_frame tensor(1, 3, H, W)
    image_frame = torch.tensor(
        img_transform({
            'image': frame
        })['image'],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    # empty the ./simulator_output/ dir
    for file in os.listdir("./simulator_output/"):
        os.remove(os.path.join("./simulator_output/", file))

    # scale_3D = 1
    # scale_3D = 10.0**-10

    video_frames = []
    
    while distance_travelled < max_distance and iter_count < max_iters:

        frame = cv2.warpAffine(frame_orig, M_2D[:2, :3], (frame.shape[1], frame.shape[0]))
        frame_merged = np.hstack([
            frame, frame, frame
        ])
        
        # Add the Z axis to the trajectory_gt
        trajectory_gt_z = np.hstack([
            traj_gt_orig, np.zeros((250, 1))
        ])

        # Apply the 3D affine transform to trajectory_gt_z
        trajectory_gt_z = np.dot(M_3D[:3, :3], trajectory_gt_z.T).T + M_3D[:3, 3]

        # Extract trajectory_gt from trajectory_gt_z
        trajectory_gt = trajectory_gt_z[:, :2]

        trajectory_gt_tensor = torch.tensor(
            trajectory_gt,
            device=device,
            dtype=torch.float32,
        ).reshape((1, 250, 2))


        # image_frame tensor(1, 3, H, W)
        # Apply 2D affine transform to image_frame
        image_frame = warp_affine_2D_torch(
            image_frame,
            M_2D_tensor,
        )

        # image_frame = torch.tensor(img_transform({
        #     'image': frame
        # })['image']).unsqueeze(0).to(device)

        trajectory_pred = net(image_frame)

        if trajectory_mode == 'templates':
            B = image_frame.shape[0]
            trajectory_pred = compute_template_trajectory(
                net,
                trajectory_pred,
                B,
                device,
            )
        
        trajectory_pred = trajectory_pred.reshape(-1, 250, 2)

        trajectory_pred_abs = relative_to_absolute(
            trajectory_pred,
            inplace=False
        )
        
        # Get the first point in the trajectory
        point_3D_topdown = trajectory_pred_abs[:, 1, :] # (1, 2) Tensor of top down view of trajectory
        
        point_3D = torch.cat(
            (
                point_3D_topdown[:, 0:1],
                point_3D_topdown[:, 1:2],
                torch.tensor([[0.0],], device=device, dtype=torch.float32),
            ),
            dim=1
        ) # (1, 3) Tensor

        distance_travelled += torch.linalg.vector_norm(
            point_3D_topdown,
            dim=1
        )


        traj_pred = trajectory_pred[0].cpu().detach().reshape((250, 2)).numpy()
        traj_gt = trajectory_gt_tensor[0].cpu().detach().reshape((250, 2)).numpy()
        frame_img = visualize_model_pred(
            traj_pred,
            traj_gt,
            frame_merged,
            K,
            D,
        )

        video_frames.append(frame_img)

        # Compute the 3D affine transform matrix
        M_3D = M_3D @ compute_3D_affine_transform(
            point_3D.detach().cpu().numpy()[0],
            M_3D
        )
        # Invert M_3D

        # Compute the 2D affine transform matrix
        M_2D = transform_3D_to_2D_flat_world(M_3D, K)
        M_2D = adjust_transformation_to_fit(M_2D, frame_orig.shape[:2])
        
        
        M_2D_tensor = torch.tensor(
            M_2D,
            device=device,
            dtype=torch.float32,
        ).reshape((1, 3, 3))

        M_3D_tensor = torch.tensor(
            M_3D,
            device=device,
            dtype=torch.float32,
        ).reshape((1, 4, 4))

        # loss is the closest distance from point_3D_topdown to the closest point in trajectory_gt
        loss += torch.min(
            torch.norm(
                point_3D_topdown - trajectory_gt_tensor,
                dim=1
            )
        )
        
        iter_count += 1

    # Print iters and distance travelled
    print("iter_count", iter_count)
    print("distance_travelled", distance_travelled)

    return loss, video_frames


def compute_3D_affine_transform(point_3D_topdown, M_3D):
    """
    point_3D_topdown: (1, 2) Tensor
    M_3D: (4, 4) numpy array
    Return:
    M_3D: (4, 4) numpy array
    """
    # Compute the 3D affine transform matrix
    M_3D = np.eye(4)
    M_3D[:3, 3] = -point_3D_topdown
    return M_3D


def transform_3D_to_2D(M_3D, K):
    """
    M_3D: (4x4) numpy array
    K: (3x3) numpy array
    Return:
    M_2D: (3x3) numpy array
    """
    K_homo = np.eye(3,4)
    K_homo[:3, :3] = K

    # Compute M_2D using the given equation
    M_2D = K_homo @ M_3D @ np.linalg.pinv(K_homo)

    return M_2D

def transform_2D_to_3D(M_2D, K):
    """
    M_3D: (4x4) numpy array
    K: (3x3) numpy array
    Return:
    M_2D: (3x3) numpy array
    """
    K_homo = np.eye(3,4)
    K_homo[:3, :3] = K

    # Compute M_2D using the given equation
    M_3D = np.linalg.pinv(K_homo) @ M_2D @ K_homo

    return M_3D

# Reload the function to transform 3D to 2D with flat-world assumption and scaling factor
def transform_3D_to_2D_flat_world(M_3D, K, scaling_factor=500.0):
    R, T = M_3D[:3, :3], M_3D[:3, 3]
    K_homo = np.eye(3, 4)
    K_homo[:3, :3] = K
    M_rot = K_homo @ M_3D @ np.linalg.pinv(K_homo)
    f_x, f_y = K[0, 0], K[1, 1]
    c_x, c_y = K[0, 2], K[1, 2]
    T_x, T_y = T[0], T[1]
    T_x *= scaling_factor
    T_y *= scaling_factor
    scale_x = 1 - T_x / f_x
    scale_y = 1 - T_y / f_y
    M_trans = np.array([[scale_x, 0, -T_x + c_x * (1 - scale_x)],
                        [0, scale_y, -T_y + c_y * (1 - scale_y)],
                        [0, 0, 1]])
    M_2D = M_trans @ M_rot
    return M_2D

# Continue with the adjustment function
def adjust_transformation_to_fit(M_2D, img_shape):
    h, w = img_shape
    corners = np.array([[0, 0, 1],
                        [w, 0, 1],
                        [w, h, 1],
                        [0, h, 1]]).T  
    new_corners_homogeneous = M_2D @ corners
    new_corners = new_corners_homogeneous[:2, :] / new_corners_homogeneous[2, :]
    min_x, min_y = np.min(new_corners, axis=1)
    max_x, max_y = np.max(new_corners, axis=1)
    trans_x = min(0, -min_x) + max(0, w - max_x)
    trans_y = min(0, -min_y) + max(0, h - max_y)
    M_trans = np.array([[1, 0, trans_x],
                        [0, 1, trans_y],
                        [0, 0, 1]])
    M_2D_adjusted = M_trans @ M_2D
    return M_2D_adjusted


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

    assert batch_size == 1, f"Simulator only supports batch_size=1, got {batch_size}"
    assert max_warp_y == 0.0, f"Simulator only supports max_warp_y=0, got {max_warp_y}"
    
    # Clear memory
    torch.cuda.empty_cache()

    # REPRODUCIBILITY
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
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

    # TODO: Validate which is the correct calibration
    # if "lwir" in dataset_name:
    #     intrinsic_matrix = np.array(
    #         [
    #             [381.19581457096, 0, 161.135532378463],
    #             [0, 382.64737644407, 138.461344582333],
    #             [0, 0, 1],
    #         ]
    #     )
    #     DistCoef = np.array(
    #         [
    #             0.0, 0.0, 0.0, 0.0, 0.0
    #         ]
    #     )

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
                        loss, video_frames = rl_sim(
                            net=net,
                            frame=frame_center[0].detach().cpu().numpy(),
                            trajectory_mode=trajectory_mode,
                            trajectory_gt=y[0].reshape(250,2).detach().cpu().numpy(),
                            img_transform=transform,
                            max_distance=0.5,
                            max_iters=5,
                            K=intrinsic_matrix,
                            D=DistCoef,
                            device=device,
                        )

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
                        evaluate_video(
                            net,
                            val_set,
                            video_frames,
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

    project_name = "{TEST}TrajDPT_V{version}_{model_type}_{dataset}_{trajectory_mode}_SIM".format(
        TEST="TEST_",
        # TEST="",
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
    wandb.agent(sweep_id, function=train_net_wandb, count=1)


if __name__ == "__main__":
    # test_rl_sim()
    # exit()
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
