import cv2
import numpy as np
from tqdm import tqdm
import torch
import wandb
# from ..datasets.autopilot_iterator import trajectory
from . import trajectory

def compute_template_trajectory(
    net,
    y_pred,
    B,
    device,
):
    
    trajectory_ohe = torch.argmax(
        y_pred,
        dim=1
    ) # (B, )
    trajectory_templates = torch.tensor(
        net.trajectory_templates
    ).to(
        device=device,
        dtype=torch.float32
    ) # (500, 250, 2) # 500 templates, each of shape (250, 2)

    # Using the one-hot encoding indices to gather the appropriate trajectories
    trajectory_absolute = torch.index_select(
        trajectory_templates,
        dim=0,
        index=trajectory_ohe
    ) # (B, 250, 2)

    trajectory_relative = trajectory.absolute_to_relative(
        trajectory_absolute
    )

    y_pred = trajectory_relative.view(
        (B, 250 * 2)
    )
    return y_pred

def visualize_model_pred(
    traj_pred,
    traj_gt,
    frame_img_merged,
    intrinsic_matrix,
    DistCoef,
):
    # traj_pred = y_pred[val_index].cpu().detach().reshape((250, 2)).numpy()
    # traj_gt = y[val_index].cpu().detach().reshape((250, 2)).numpy()

    # frame_img_orig = frame_center[val_index].cpu().detach().numpy()
    # frame_img_merged = frame_merged[val_index].cpu().detach().numpy()
    frame_img = frame_img_merged[
        0 : frame_img_merged.shape[0],
        frame_img_merged.shape[1]
        // 3 : frame_img_merged.shape[1]
        // 3
        * 2,
        :,
    ]
    trajectory.plot_steering_traj(
        frame_img,
        trajectory.trajectory_rel_2_trajectory(traj_gt),
        color=(0, 255, 0),
        intrinsic_matrix=intrinsic_matrix,
        DistCoef=DistCoef,
    )

    trajectory.plot_steering_traj(
        frame_img,
        trajectory.trajectory_rel_2_trajectory(traj_pred),
        color=(0, 0, 255),
        intrinsic_matrix=intrinsic_matrix,
        DistCoef=DistCoef,
    )

    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    return frame_img

@torch.no_grad()
def evaluate(net, val_set, trajectory_mode, device, amp, global_step, epoch, experiment, intrinsic_matrix, DistCoef, lr):
    histograms = {}
    for tag, value in net.named_parameters():
        if value is not None and value.grad is not None:
            tag = tag.replace("/", ".")
            histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
            histograms["Gradients/" + tag] = wandb.Histogram(
                value.grad.data.cpu()
            )
    metrics_l = {
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'MAPE': [],
        'Cosine Similarity': [],
        'DTW Distance': [],
    }
    log_images = []
    net.eval()
    # for batch_index in range(0, len(val_set), 1):
    for batch_index in tqdm(
        # range(0, min(3, len(val_set)), 1),
        range(len(val_set)),
        desc="Evaluation"
    ):
        batch = get_batch(val_set, batch_index, batch_size=1, N=4)

        x, frame_merged, frame_center, y = batch
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=amp):
            y_pred = net(x)

        if trajectory_mode == 'templates':
            B = x.shape[0]
            y_pred = compute_template_trajectory(
                net,
                y_pred,
                B,
                device,
            )

        # Compute RMSE
        metrics = trajectory.trajectory_metrics(
            y, y_pred
        )
        for m in metrics:
            metrics_l[m] += [metrics[m], ]


        if batch_index >= 3:
            continue
        
        val_index = 0

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

        log_images.append(frame_img)

    log_images = np.concatenate(log_images, axis=1)
    
    for m in metrics_l:
        metrics_l[m] = np.mean(np.array(metrics_l[m]))

    experiment.log(
        {
            "learning rate": lr,
            "plot": wandb.Image(log_images),
            "step": global_step,
            "epoch": epoch,
            **metrics_l,
            **histograms,
        }
    )

    net.train()

@torch.no_grad()
def evaluate_video(net, val_set, video_frames, trajectory_mode, device, amp, global_step, epoch, experiment, intrinsic_matrix, DistCoef, lr):
    histograms = {}
    for tag, value in net.named_parameters():
        if value is not None and value.grad is not None:
            tag = tag.replace("/", ".")
            histograms["Weights/" + tag] = wandb.Histogram(value.data.cpu())
            histograms["Gradients/" + tag] = wandb.Histogram(
                value.grad.data.cpu()
            )
    metrics_l = {
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'MAPE': [],
        'Cosine Similarity': [],
        'DTW Distance': [],
    }
    log_images = []
    net.eval()
    # for batch_index in range(0, len(val_set), 1):
    for batch_index in tqdm(
        # range(0, min(3, len(val_set)), 1),
        range(len(val_set)),
        desc="Evaluation"
    ):
        batch = get_batch(val_set, batch_index, batch_size=1, N=4)

        x, frame_merged, frame_center, y = batch
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=amp):
            y_pred = net(x)

        if trajectory_mode == 'templates':
            B = x.shape[0]
            y_pred = compute_template_trajectory(
                net,
                y_pred,
                B,
                device,
            )

        # Compute RMSE
        metrics = trajectory.trajectory_metrics(
            y, y_pred
        )
        for m in metrics:
            metrics_l[m] += [metrics[m], ]


        if batch_index >= 3:
            continue
        
        val_index = 0

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

        log_images.append(frame_img)

    log_images = np.concatenate(log_images, axis=1)
    video_frames = np.concatenate([[i.transpose(2,0,1),] for i in video_frames], axis=0)


    for m in metrics_l:
        metrics_l[m] = np.mean(np.array(metrics_l[m]))

    experiment.log(
        {
            "learning rate": lr,
            "plot": wandb.Image(log_images),
            "simulation": wandb.Video(video_frames, fps=1),
            "step": global_step,
            "epoch": epoch,
            **metrics_l,
            **histograms,
        }
    )

    net.train()


def get_batch(train_set, batch_index, batch_size, N=4):
    batch = []
    for _ in range(N):
        batch += [[]]
    for sub_index in range(batch_index - batch_size, batch_index, 1):
        new_batch = train_set[sub_index]
        for sub_cat in range(len(batch)):
            batch[sub_cat] += [new_batch[sub_cat]]

    for sub_cat in range(len(batch)):
        batch[sub_cat] = torch.cat(batch[sub_cat], dim=0)

    return batch

from torchvision import transforms
from PIL import Image

def img_transform(img, unsqueeze=False):
    tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize
    img = cv2.resize(img, (640, 480))
    img = Image.fromarray(img)

    # img = np.transpose(img, (2, 0, 1))
    img = tfms(img)
    img = img.permute(2, 0, 1)

    if unsqueeze:
        img = img.unsqueeze(0)

    # transpose (B, C, H, W)
    

    return img
