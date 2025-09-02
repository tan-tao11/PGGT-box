from typing import Dict
import torch
import math
import torch.nn.functional as F

def compute_camera_loss(
        pred_bbox_encodings,
        pred_confs,
        batch_data,
        gamma=0.6,
        config=None,
):
    """
    Compute camera loss from all stages
    """
    # Recode losses for logging
    loss_dict = {
        "total_loss": 0,
        "loss_bbox": 0,
        "loss_conf": 0,
    }
    # Number of prediction stages
    n_stages = len(pred_bbox_encodings)

    # Get ground truth camera pose encodings
    gt_bbox_encodings = batch_data['bbox_2d_offset_label'].to(pred_bbox_encodings[0].device)

    # Initialize loss accumulators for translation and rotation
    total_loss_bbox = total_loss_conf = 0

    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_bbox_stage = pred_bbox_encodings[stage_idx].float()
        pred_conf_stage = pred_confs[stage_idx].float()

        loss_bbox_stage, loss_conf_stage = \
            loss_single_stage(pred_bbox_stage, pred_conf_stage, gt_bbox_encodings, batch_data, config, loss_dict)
        
        total_loss_bbox += stage_weight * loss_bbox_stage
        total_loss_conf += stage_weight * loss_conf_stage

    # Average over all stages
    avg_loss_bbox = total_loss_bbox / n_stages
    avg_loss_conf = total_loss_conf / n_stages
    total_loss = avg_loss_bbox + avg_loss_conf

    for k, v in loss_dict.items():
        loss_dict[k] = v / n_stages
    loss_dict['total_loss'] = total_loss.detach()

    return total_loss, loss_dict

def loss_single_stage(pred_bboxes, pred_confs, gt_bboxes, batch_data, config, loss_dict=None):
    """
    Compute loss for single stage
    args:
        pred_bboxes: (B, N, 16)
        pred_confs: (B, N, 8)
        gt_poses: (B, N, 9)
    """

    # Bbox: L1 loss
    bbox_loss_individual = torch.sum(torch.abs(pred_bboxes - gt_bboxes), dim=-1)  # (B, N)
    bbox_loss = torch.mean(bbox_loss_individual) * config.train.loss_weights.weight_bbox

    query_info = batch_data['query_info']
    query_pose = query_info['query_pose'].to(pred_bboxes.device)
    ref_info = batch_data['ref_info']
    ref_poses = ref_info['ref_poses'].to(pred_bboxes.device)

    # Compute relative rotation angle between query and ref poses
    angles = relative_rotation_angle_torch(ref_poses[..., :3, :3], query_pose[..., :3, :3])

    '''
    # Conf: KL loss  
    # Smaller angles are more confident
    T = config.train.conf_T
    gt_conf = torch.softmax(-(angles.detach()/T), dim=1)
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # log-prob vs prob
    pred_confs = pred_confs[:, 1:, 0]  # (B, N-1)
    log_pred_conf = torch.log_softmax(pred_confs, dim=1)  # log-softmax
    confidence_loss = kl_loss_fn(log_pred_conf, gt_conf) * config.train.loss_weights.weight_conf

    # Conf: L1 loss
    pred_conf_softmax = torch.softmax(pred_confs, dim=1)
    confidence_loss += torch.mean(torch.abs(pred_conf_softmax - gt_conf)) * config.train.loss_weights.weight_conf
    '''

    # Conf: Cross entropy loss
    scale = 10.0 
    pred_confs_ref = pred_confs[:, 1:, 0] * scale  # (B, N-1)
    confidence_loss = F.cross_entropy(pred_confs_ref, torch.argmin(angles, dim=1)) * config.train.loss_weights.weight_conf

    if loss_dict is not None:
        loss_dict['loss_bbox'] += bbox_loss.detach()
        loss_dict['loss_conf'] += confidence_loss.detach()

    return bbox_loss, confidence_loss

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

# def loss_single_stage(pred_poses, pred_confs, gt_poses, config, loss_dict=None):
#     """
#     Compute loss for single stage
#     args:
#         pred_poses: (B, N, 9)
#         pred_confs: (B, N, 2)
#         gt_poses: (B, N, 9)
#     """
#     # 预测部分
#     pred_rot_6d = pred_poses[..., 3:]  # (B, N, 6)
#     pred_trans = pred_poses[..., :3]   # (B, N, 3)

#     # 真值部分
#     gt_rot_6d = gt_poses[..., 3:]      # (B, N, 6)
#     gt_trans = gt_poses[..., :3]       # (B, N, 3)
    
#     # --- 2. 计算平移损失 (L1 Loss) ---
#     # L1损失对异常值更鲁棒。gt_trans会自动广播到 (B, N, 3)
#     center_loss = torch.sum(torch.abs(pred_trans[..., :2] - gt_trans[..., :2]), dim=-1)  # (B, N)
#     depth_loss = torch.abs(pred_trans[..., 2] - gt_trans[..., 2])  # (B, N)
#     loss_t_individuals = center_loss * config.train.loss_weights.weight_center + depth_loss * config.train.loss_weights.weight_depth
#     loss_t = torch.mean(loss_t_individuals)

#     # --- 3. 计算旋转损失 (Geodesic Distance) ---
#     # 将6D表示转换为旋转矩阵
#     pred_rot_mat = rotation_6d_to_matrix(pred_rot_6d)  # (B, N, 3, 3)
#     gt_rot_mat = rotation_6d_to_matrix(gt_rot_6d)      # (B, N, 3, 3)

#     relative_rot_mat = torch.matmul(pred_rot_mat.float(), gt_rot_mat.transpose(-2, -1))
#     # 使用einsum高效计算批量矩阵的迹
#     trace = torch.einsum('bnii->bn', relative_rot_mat) # (B, N)
    
#     # clip防止由于浮点数误差导致值超出[-1, 1]范围
#     cos_theta = (trace - 1) / 2
    
#     # rot_loss = torch.acos(cos_theta) # (B, N), 结果是弧度制的角误差
#     rot_loss = 1 - cos_theta
#     loss_R_individuals = rot_loss * config.train.loss_weights.weight_rot
#     loss_R = torch.mean(loss_R_individuals)

#     # --- 4. 计算置信度损失 ---
#     # 计算平移置信度真值
#     T = config.train.conf_T
#     gt_conf_t = torch.softmax(-loss_t_individuals[:, 1:].detach()/T, dim=1)

#     # Compute translation confidence loss
#     kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # log-prob vs prob
#     pred_confs_t = pred_confs[:, 1:, 0]  # (B, N-1)
#     log_pred_conf_t = torch.log_softmax(pred_confs_t, dim=1)  # log-softmax
#     confidence_t_loss = kl_loss_fn(log_pred_conf_t, gt_conf_t) * config.train.loss_weights.weight_conf_t

#     # Get ground truth rotation confidence
#     gt_conf_R = torch.softmax(-loss_R_individuals[:, 1:].detach()/T, dim=1)

#     # Compute rotation confidence loss
#     pred_confs_R = pred_confs[:, 1:, 1]  # (B, N-1)
#     log_pred_conf_R = torch.log_softmax(pred_confs_R, dim=1)  # log-softmax
#     confidence_R_loss = kl_loss_fn(log_pred_conf_R, gt_conf_R) * config.train.loss_weights.weight_conf_R

#     if loss_dict is not None:
#         loss_dict['loss_center'] += torch.mean(center_loss.detach())
#         loss_dict['loss_depth'] += torch.mean(depth_loss.detach())
#         loss_dict['loss_rot'] += torch.mean(loss_R.detach())
#         loss_dict['loss_conf_t'] += confidence_t_loss.detach()
#         loss_dict['loss_conf_R'] += confidence_R_loss.detach()

#     return loss_t, loss_R, confidence_t_loss, confidence_R_loss


def cal_loss_with_confidence(preds_with_conf, gt, config, is_train=False):
    # preds_with_conf : (pred_pose, conf_logits)]
    loss_dict = {}

    gt = gt
    # 将N个预测结果堆叠起来
    all_pred_poses = preds_with_conf[0].float() #  (B, N, 9)
    all_pred_conf_logits = preds_with_conf[1].float()  # (B, N, 1)

    # --- 1. 计算每个预测的姿态损失 ---
    # 这里的 pose_loss_fn 是你之前计算旋转和平移误差的函数
    # 我们需要它能计算一个批次中每个预测的单独损失
    # 假设 pose_loss_fn 返回每个预测的损失值，形状为 (B, N)
    individual_pose_losses, rot_loss, _, _ = pose_loss_fn(loss_dict,
                                        all_pred_poses, 
                                        gt, 
                                        rot_weight=config.train.loss_weights.weight_rot, 
                                        center_weight=config.train.loss_weights.weight_center, 
                                        depth_weight=config.train.loss_weights.weight_depth
                                        ) # (B, N)

    # 【新方法：生成软标签】
    with torch.no_grad():
        T = config.train.conf_T
        # confidence_gt = torch.softmax(-individual_pose_losses/T, dim=1).unsqueeze(-1)
        confidence_gt = torch.softmax(-rot_loss[:, 1:]/T, dim=1)

    # 【新方法】
    # --- 4. 计算置信度损失 ---
    # 先对logits使用sigmoid激活函数，得到[0,1]的概率
    pred_conf = torch.sigmoid(all_pred_conf_logits)

    if config.train.pose_loss_type == "average":
        average_pose_loss = individual_pose_losses.mean()
        pose_loss = average_pose_loss
        # 使用 L1 Loss
        l1_loss_fn = torch.nn.L1Loss()
        confidence_loss = l1_loss_fn(pred_conf, confidence_gt)
    elif config.train.pose_loss_type == "best":
        # 【新方法：只计算最佳预测的损失】
        with torch.no_grad():
            best_pred_indices = torch.argmin(individual_pose_losses[:, 1:], dim=1) # (B,)
        best_pose_loss = individual_pose_losses[:, 1:].gather(1, best_pred_indices.unsqueeze(-1)).mean()
        # 第1副图像的位姿损失
        query_pose_loss = individual_pose_losses[:, 0].mean()
        pose_loss = best_pose_loss + query_pose_loss # 使用这个作为姿态损失
        loss_dict['best_pose_loss'] = best_pose_loss
        loss_dict['query_pose_loss'] = query_pose_loss
        # 使用 L1 Loss
        l1_loss_fn = torch.nn.L1Loss()
        confidence_loss = l1_loss_fn(pred_conf, confidence_gt)
    elif config.train.pose_loss_type == "weight":
        # ===  使用 pred conf logits 计算 softmax 权重 ===
        # all_pred_conf_logits: shape (B, N)
        # confidence_weight = torch.softmax(all_pred_conf_logits, dim=1)  # (B, N, 1)
        # === 4. 计算置信度加权 pose loss ===
        # pose_loss = torch.sum(confidence_weight * individual_pose_losses[..., None], dim=1).mean()
        weighted_pose_loss = torch.sum(confidence_gt * individual_pose_losses[:, 1:], dim=1).mean()
        query_pose_loss = individual_pose_losses[:, 0].mean()
        pose_loss = weighted_pose_loss + query_pose_loss # 使用这个作为姿态损失
        # ===  置信度监督项（L1 Loss）===
        # pred_conf = torch.sigmoid(all_pred_conf_logits)  # (B, N)
        # l1_loss_fn = torch.nn.L1Loss()
        # confidence_l1_loss = l1_loss_fn(pred_conf, confidence_gt)

        # === 可选增强：KL散度（用于匹配置信度分布）===
        kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')  # log-prob vs prob
        log_pred_conf = torch.log_softmax(all_pred_conf_logits[:, 1:, 0], dim=1)  # log-softmax
        confidence_kl_loss = kl_loss_fn(log_pred_conf, confidence_gt)

        # confidence_loss = config.train.loss_weights.weight_conf_l1 * confidence_l1_loss + config.train.loss_weights.weight_conf_kl * confidence_kl_loss
        confidence_loss = config.train.loss_weights.weight_conf_kl * confidence_kl_loss
        loss_dict["weighted_pose_loss"] = weighted_pose_loss.detach()
        loss_dict['query_pose_loss'] = query_pose_loss.detach()

    # --- 6. 组合总损失 ---
    # lambda_conf 是一个超参数，用于平衡两个损失，例如可以设为0.1
    total_loss = pose_loss + config.train.loss_weights.weight_conf * confidence_loss

    loss_dict["total_loss"] = total_loss
    loss_dict["confidence_loss"] = confidence_loss.detach()

    # 返回一个字典，方便在Tensorboard中观察
    return loss_dict

def pose_loss_fn(loss_dict: Dict, pred_poses: torch.Tensor, gt_pose: torch.Tensor, rot_weight: float = 1.0, center_weight: float = 1.0, depth_weight: float = 1.0) -> torch.Tensor:
    """
    计算每个预测姿态与真值姿态之间的损失。
    旋转部分使用几何上的角误差，平移部分使用L1误差。

    Args:
        pred_poses (torch.Tensor): 模型的N个姿态预测，形状为 (B, N, 9)。
        gt_pose (torch.Tensor): 真值姿态，形状为 (B, 1, 9)，会自动广播。
        rot_weight (float): 旋转损失的权重。
        trans_weight (float): 平移损失的权重。

    Returns:
        torch.Tensor: 每个预测的单独总损失，形状为 (B, N)。
    """
    # --- 1. 将预测和真值分解为旋转和平移部分 ---
    # 预测部分
    pred_rot_6d = pred_poses[..., 3:]  # (B, N, 6)
    pred_trans = pred_poses[..., :3]   # (B, N, 3)

    # 真值部分
    gt_rot_6d = gt_pose[..., 3:]      # (B, N, 6)
    gt_trans = gt_pose[..., :3]       # (B, N, 3)
    
    # --- 2. 计算平移损失 (L1 Loss) ---
    # L1损失对异常值更鲁棒。gt_trans会自动广播到 (B, N, 3)
    center_loss = torch.sum(torch.abs(pred_trans[..., :2] - gt_trans[..., :2]), dim=-1)  # (B, N)
    depth_loss = torch.abs(pred_trans[..., 2] - gt_trans[..., 2])  # (B, N)

    # --- 3. 计算旋转损失 (Geodesic Distance) ---
    # 将6D表示转换为旋转矩阵
    pred_rot_mat = rotation_6d_to_matrix(pred_rot_6d)  # (B, N, 3, 3)
    gt_rot_mat = rotation_6d_to_matrix(gt_rot_6d)      # (B, N, 3, 3)

    # 计算两个旋转矩阵之间的相对旋转
    # R_err = R_pred * R_gt^T (R_gt的逆就是其转置)
    # gt_rot_mat会自动广播到 (B, N, 3, 3)
    relative_rot_mat = torch.matmul(pred_rot_mat.float(), gt_rot_mat.transpose(-2, -1))

    # 从相对旋转矩阵计算角误差（弧度）
    # trace(R) = 1 + 2*cos(theta) -> theta = acos((trace(R) - 1) / 2)
    # 使用einsum高效计算批量矩阵的迹
    trace = torch.einsum('bnii->bn', relative_rot_mat) # (B, N)
    
    # clip防止由于浮点数误差导致值超出[-1, 1]范围
    cos_theta = (trace - 1) / 2
    
    # rot_loss = torch.acos(cos_theta) # (B, N), 结果是弧度制的角误差
    rot_loss = 1 - cos_theta
    if loss_dict is not None:
        loss_dict['rot_loss'] = torch.mean(rot_loss.detach())
        loss_dict['center_loss'] = torch.mean(center_loss.detach()) 
        loss_dict['depth_loss'] = torch.mean(depth_loss.detach()) 
    
    # --- 4. 组合损失 ---
    # 使用权重来平衡旋转和平移损失的量级
    total_individual_loss = rot_weight * rot_loss + center_weight * center_loss + depth_weight * depth_loss
    
    return total_individual_loss, rot_loss, center_loss, depth_loss

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

# def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
#     """
#     Converts batched 6D rotation representation to 3x3 rotation matrices.

#     Args:
#         d6 (torch.Tensor): Rotation in 6D representation, shape (B, N, 6)

#     Returns:
#         torch.Tensor: Rotation matrices, shape (B, N, 3, 3)
#     """
#     # Split into two 3D vectors
#     a1 = d6[..., 0:3]  # shape: (B, N, 3)
#     a2 = d6[..., 3:6]  # shape: (B, N, 3)

#     # Normalize a1 to get the first basis vector
#     b1 = F.normalize(a1, dim=-1)  # shape: (B, N, 3)

#     # Make a2 orthogonal to b1
#     dot_prod = torch.sum(b1 * a2, dim=-1, keepdim=True)  # shape: (B, N, 1)
#     a2_ortho = a2 - dot_prod * b1
#     b2 = F.normalize(a2_ortho, dim=-1)

#     # b3 = b1 x b2
#     b3 = torch.cross(b1, b2, dim=-1)

#     # Stack into rotation matrix
#     rot_mat = torch.stack((b1, b2, b3), dim=-1)  # shape: (B, N, 3, 3)

#     return rot_mat

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts batched 6D rotation representation to 3x3 rotation matrices.

    Args:
        d6 (torch.Tensor): Rotation in 6D representation, shape (B, N, 6)

    Returns:
        torch.Tensor: Rotation matrices, shape (B, N, 3, 3)
    """
    # Define a small epsilon for numerical stability
    eps = 1e-8

    # Split into two 3D vectors
    a1 = d6[..., 0:3]  # shape: (B, N, 3)
    a2 = d6[..., 3:6]  # shape: (B, N, 3)

    # Normalize a1 to get the first basis vector
    # MODIFIED: Added eps to prevent division by zero
    b1 = F.normalize(a1, dim=-1, eps=eps)  # shape: (B, N, 3)

    # Make a2 orthogonal to b1
    dot_prod = torch.sum(b1 * a2, dim=-1, keepdim=True)  # shape: (B, N, 1)
    a2_ortho = a2 - dot_prod * b1
    # MODIFIED: Added eps to prevent division by zero
    b2 = F.normalize(a2_ortho, dim=-1, eps=eps)

    # b3 = b1 x b2
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into rotation matrix
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # shape: (B, N, 3, 3)

    return rot_mat