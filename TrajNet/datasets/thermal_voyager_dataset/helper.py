import matplotlib as mpl
import numpy as np
import torch

# device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths"""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_masked_errors(gt, pred, mask):
    """
    Computation of error metrics between predicted and ground truth depths only for the masked region

    Args:
    gt: numpy array of shape (H, W) representing the ground truth depths
    pred: numpy array of shape (H, W) representing the predicted depths
    mask: numpy array of shape (H, W) representing the mask indicating the region of interest

    Returns:
    tuple of error metrics for the masked region: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    masked_gt = gt[mask]
    masked_pred = pred[mask]

    thresh = np.maximum((masked_gt / masked_pred), (masked_pred / masked_gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25**2).mean()
    a3 = (thresh < 1.25**3).mean()

    rmse = (masked_gt - masked_pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if np.isinf(rmse) or np.isnan(rmse):
        rmse = 0

    rmse_log = (
        np.log(masked_gt) - np.log(masked_pred)
    ) ** 2  # RuntimeWarning: invalid value encountered in log
    rmse_log = np.sqrt(rmse_log.mean())
    if np.isinf(rmse_log) or np.isnan(rmse_log):
        rmse_log = 0

    abs_rel = np.mean(np.abs(masked_gt - masked_pred) / masked_gt)
    if np.isinf(abs_rel) or np.isnan(abs_rel):
        abs_rel = 0

    sq_rel = np.mean(((masked_gt - masked_pred) ** 2) / masked_gt)
    if np.isinf(sq_rel) or np.isnan(sq_rel):
        sq_rel = 0

    a1 = 0 if np.isnan(a1) else a1
    a2 = 0 if np.isnan(a2) else a2
    a3 = 0 if np.isnan(a3) else a3

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


##############################################################
# Color map logic

cmap = mpl.colormaps["viridis"]
# cmap = mpl.colormaps['magma']

colors_hash = []
colors_hash_res = 256
for i in range(0, colors_hash_res):
    colors_hash.append(cmap(float(i) / (colors_hash_res - 1)))


def color_by_index(
    POINTS_np, index=2, invert=False, min_height=None, max_height=None
):
    if POINTS_np.shape[0] == 0:
        return np.ones_like(POINTS_np)
    heights = POINTS_np[:, index].copy()
    heights_filter = np.logical_not(
        np.logical_and(np.isnan(heights), np.isinf(heights))
    )
    if max_height is None:
        max_height = np.max(heights[heights_filter])
    if min_height is None:
        min_height = np.min(heights[heights_filter])
    # heights = np.clip(heights, min_height, max_height)
    heights = (heights - min_height) / (max_height - min_height)
    if invert:
        heights = 1.0 - heights
    # heights[np.logical_not(heights_filter)] = 0.0
    heights = np.clip(heights, 0.0, 1.0)
    heights_color_index = np.rint(heights * (colors_hash_res - 1)).astype(
        np.uint8
    )

    COLORS_np = np.array([colors_hash[xi] for xi in heights_color_index])
    return (COLORS_np * 255).astype(np.uint8)


##############################################################
