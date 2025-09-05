import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_loss(pred_pv_map_offset, pred_confs, data, config):
    loss_dict = {}
    # 计算向量图损失
    gt_pv_map_offset = data['pv_maps_offset'].to(pred_pv_map_offset.device)
    loss_fn = nn.SmoothL1Loss(reduction='none', beta=config.train.smooth_l1_beta)  # Smooth L1 Loss
    loss_pv_map = loss_fn(pred_pv_map_offset, gt_pv_map_offset)
    loss_pv_map = loss_pv_map.mean()  # 平均所有元素
    loss_dict['loss_pv_map'] = loss_pv_map.detach()
    
    # 计算置信度损失
    query_info = data['query_info']
    query_pose = query_info['query_pose'].to(pred_pv_map_offset.device)
    ref_info = data['ref_info']
    ref_poses = ref_info['ref_poses'].to(pred_pv_map_offset.device)
    # Compute relative rotation angle between query and ref poses
    angles = relative_rotation_angle_torch(ref_poses[..., :3, :3], query_pose[..., :3, :3])
    # Conf: Cross entropy loss
    scale = 10.0 
    pred_confs_ref = pred_confs[:, 1:, 0] * scale  # (B, N-1)
    confidence_loss = F.cross_entropy(pred_confs_ref, torch.argmin(angles, dim=1)) * config.train.loss_weights.weight_conf
    loss_dict['loss_conf'] = confidence_loss.detach()
    
    total_loss = loss_pv_map * config.train.loss_weights.weight_pv_map + confidence_loss * config.train.loss_weights.weight_conf
    loss_dict['total_loss'] = total_loss.detach()
    
    return total_loss, loss_dict


def relative_rotation_angle_torch(ref_rots, tgt_rots, degrees=False, eps=1e-7):
    """
    ref_rots: tensor (B, N, 3, 3)
    tgt_rots: tensor (B, 1, 3, 3) or (B, 3, 3)
    returns: tensor angles (B, N) in radians by default (degrees if degrees=True)
    """
    if tgt_rots.dim() == 3:
        tgt_rots = tgt_rots.unsqueeze(1)  # -> (B,1,3,3)

    # R_rel = R_tgt @ R_ref^T  -> shape (B, N, 3, 3)
    R_rel = torch.matmul(tgt_rots, ref_rots.transpose(-2, -1))

    # trace: (B, N)
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]

    # cos(theta) = (trace - 1) / 2, clamp for numerical stability
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0 + eps, 1.0 - eps)

    angles = torch.acos(cos_theta)  # (B, N) in radians

    if degrees:
        angles = angles * (180.0 / math.pi)

    return angles