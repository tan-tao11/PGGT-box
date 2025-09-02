import torch
import torch.nn.functional as F

def cal_loss(pred, gt, config=None, is_train=False):
    pred_trans = pred[:, :, :3]
    pred_rot = pred[:, :, 3:]
    gt_trans = gt[:, :, :3]
    gt_rot = gt[:, :, 3:]

    # trans_losss = trans_losss_l1(pred_trans, gt_trans)
    center_loss, depth_loss = trans_center_depth_loss(pred_trans, gt_trans)
    rot_loss = rot_loss_angle(pred_rot, gt_rot)
    # if is_train:
    #     loss = {
    #         'trans_loss': trans_losss * config.train.loss_weights.weight_trans,
    #         'rot_loss': rot_loss * config.train.loss_weights.weight_rot
    #     }
    # else:
    #     loss = {
    #         'trans_loss': trans_losss,
    #         'rot_loss': rot_loss
    #     }
    if is_train:
        loss = {
            'center_loss': center_loss * config.train.loss_weights.weight_center,
            'depth_loss': depth_loss * config.train.loss_weights.weight_depth,
            'rot_loss': rot_loss * config.train.loss_weights.weight_rot
        }
    else:
        loss = {
            'center_loss': center_loss,
            'depth_loss': depth_loss,
            'rot_loss': rot_loss
        }
    return loss

def trans_center_depth_loss(pred, gt):
    pred_center = pred[:, :, :2]
    gt_center = gt[:, :, :2]
    loss_center = F.l1_loss(pred_center, gt_center)

    pred_depth = pred[:, :, 2:3]
    gt_depth = gt[:, :, 2:3]
    loss_depth = F.l1_loss(pred_depth, gt_depth)

    return loss_center, loss_depth

def trans_losss_l1(pred, gt):
    loss = torch.nn.functional.l1_loss(pred, gt)
    return loss

def rot_loss_l1(pred, gt):
    loss = torch.nn.functional.l1_loss(pred, gt)
    return loss

def rot_loss_angle(pred, gt):
    pre_mat = rotation_6d_to_matrix(pred)
    gt_mat = rotation_6d_to_matrix(gt)
    # loss = rotation_matrix_angle_error(pre_mat, gt_mat)
    loss = cosine_rotation_loss(pre_mat, gt_mat)

    return torch.mean(loss)

def rotation_matrix_angle_error(R1: torch.Tensor, R2: torch.Tensor, return_degree: bool = False) -> torch.Tensor:
    """
    Compute angular error (in degrees or radians) between rotation matrices.

    Args:
        R1 (torch.Tensor): Predicted rotation matrices of shape (..., 3, 3)
        R2 (torch.Tensor): Ground truth rotation matrices of shape (..., 3, 3)
        return_degree (bool): If True, return error in degrees; else radians.

    Returns:
        torch.Tensor: Angular error of shape (..., )
    """
    assert R1.shape == R2.shape and R1.shape[-2:] == (3, 3), "Input shapes must be (..., 3, 3)"
    
    # Compute relative rotation matrix
    R_rel = R1 @ R2.transpose(-2, -1)  # shape (..., 3, 3)

    # Compute trace of relative rotation
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]  # shape (...)

    # Clamp to avoid numerical issues (acos input must be in [-1, 1])
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Angle in radians
    angle = torch.acos(cos_theta)

    if return_degree:
        angle = angle * (180.0 / torch.pi)

    return angle

def geodesic_loss(R1, R2):
    """
    Compute geodesic loss (more stable than acos-based angle loss)
    """
    R_rel = R1 @ R2.transpose(-2, -1)
    log_R = torch.linalg.logm(R_rel)  # matrix log, may be slow
    return torch.norm(log_R, dim=(-2, -1)) / (2 ** 0.5)

def cosine_rotation_loss(R1, R2):
    """
    Approximate rotation loss using trace-based cosine distance
    """
    R_rel = R1 @ R2.transpose(-2, -1)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    loss = 1 - (trace - 1) / 2  # 1 - cosθ
    return loss.mean()

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts batched 6D rotation representation to 3x3 rotation matrices.

    Args:
        d6 (torch.Tensor): Rotation in 6D representation, shape (B, N, 6)

    Returns:
        torch.Tensor: Rotation matrices, shape (B, N, 3, 3)
    """
    # Split into two 3D vectors
    a1 = d6[..., 0:3]  # shape: (B, N, 3)
    a2 = d6[..., 3:6]  # shape: (B, N, 3)

    # Normalize a1 to get the first basis vector
    b1 = F.normalize(a1, dim=-1)  # shape: (B, N, 3)

    # Make a2 orthogonal to b1
    dot_prod = torch.sum(b1 * a2, dim=-1, keepdim=True)  # shape: (B, N, 1)
    a2_ortho = a2 - dot_prod * b1
    b2 = F.normalize(a2_ortho, dim=-1)

    # b3 = b1 x b2
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into rotation matrix
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # shape: (B, N, 3, 3)

    return rot_mat