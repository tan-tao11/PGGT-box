import torch
import torch.nn as nn
import torch.nn.functional as F

# def cal_loss(pred_pv_map_offset, pred_confs, data, config):
#     loss_dict = {}
#     # 计算向量图损失
#     gt_pv_map_offset = data['pv_maps_offset'].to(pred_pv_map_offset.device)
#     loss_fn = nn.SmoothL1Loss(reduction='none', beta=config.train.smooth_l1_beta)  # Smooth L1 Loss
#     loss_pv_map = loss_fn(pred_pv_map_offset, gt_pv_map_offset)
#     loss_pv_map = loss_pv_map.mean()  # 平均所有元素
#     loss_dict['loss_pv_map'] = loss_pv_map.detach()

#     # 计算置信度损失
#     query_info = data['query_info']
#     query_pose = query_info['query_pose'].to(pred_pv_map_offset.device)
#     ref_info = data['ref_info']
#     ref_poses = ref_info['ref_poses'].to(pred_pv_map_offset.device)
    
#     # Compute relative rotation angle between query and ref poses
#     angles = relative_rotation_angle_torch(ref_poses[..., :3, :3], query_pose[..., :3, :3])
#     # Conf: Cross entropy loss
#     # scale = 10.0 
#     # pred_confs_ref = pred_confs[:, 1:, 0] * scale  # (B, N-1)
#     # confidence_loss = F.cross_entropy(pred_confs_ref, torch.argmin(angles, dim=1))
#     # loss_dict['loss_conf'] = confidence_loss.detach()

#     # 回归损失
#     temperature = 1.0 
#     confidence_gt = torch.exp(-angles / temperature) 
#     pred_confs_ref = torch.sigmoid(pred_confs[:, 1:, 0])
#     confidence_loss = F.l1_loss(pred_confs_ref, confidence_gt) 

#     loss_dict['loss_conf'] = confidence_loss.detach()

#     total_loss = loss_pv_map * config.train.loss_weights.weight_pv_map + confidence_loss * config.train.loss_weights.weight_conf
#     loss_dict['total_loss'] = total_loss.detach()
    
#     return total_loss, loss_dict

def cal_loss(pred_pv_map_offset, pred_confs, data, config):
    loss_dict = {}
    
    # --- 向量图损失部分 ---
    gt_pv_map_offset = data['pv_maps_offset'].to(pred_pv_map_offset.device)
    
    # 保持 reduction='none' 以便我们能得到每个元素的损失
    loss_fn = nn.SmoothL1Loss(reduction='none', beta=config.train.smooth_l1_beta)
    
    # loss_per_element 的形状与输入相同, e.g., (B, N_ref-1, H, W, C)
    loss_per_element = loss_fn(pred_pv_map_offset, gt_pv_map_offset)

    query_info = data['query_info']
    query_pose = query_info['query_pose'].to(pred_pv_map_offset.device)
    ref_info = data['ref_info']
    ref_poses = ref_info['ref_poses'].to(pred_pv_map_offset.device)
    
    # Compute relative rotation angle between query and ref poses
    angles = relative_rotation_angle_torch(ref_poses[..., :3, :3], query_pose[..., :3, :3])

    # angles 的形状: (B, N-1)
    # temperature 用于控制权重的“尖锐”程度。值越小，权重差异越大。
    temperature = 0.1 # 这是一个需要调整的超参数

    # 注意要在angles前加负号，因为我们希望angle小的权重高
    weights = torch.softmax(-angles / temperature, dim=-1)

    # (重要) 为避免权重本身参与梯度计算，最好detach()
    weights = weights.detach()

    # 【修改点 1】: 计算“逐样本”的向量图损失
    # 我们需要为 B * (N_ref-1) 个样本中的每一个都计算一个损失均值
    # 因此，我们在 H, W, C 这几个维度上求平均
    # 得到的 per_sample_loss 形状为 (B, N_ref-1)
    # loss_per_element_weight = loss_per_element
    # loss_per_element_weight[:, 1:] = loss_per_element_weight[:, 1:] * weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
    per_sample_loss = loss_per_element.mean(dim=(-3, -2, -1))
    per_sample_loss_weight = per_sample_loss.clone()
    per_sample_loss_weight[:, 1:] = per_sample_loss_weight[:, 1:] * weights
    # 整个batch的向量图损失，用于反向传播
    # 注意：我们使用 per_sample_loss 计算均值，而不是 loss_per_element
    # 这样可以确保梯度流是正确的
    loss_pv_map = per_sample_loss_weight.mean()
    loss_dict['loss_pv_map'] = loss_pv_map.detach()

    # --- 置信度损失部分 ---
    
    # 【修改点 2】: 【关键步骤】使用 .detach() 切断梯度流
    # 我们用 per_sample_loss 来生成标签，但这个过程不能影响原有的梯度计算
    # per_sample_loss_detached 不再带有梯度信息
    per_sample_loss_detached = per_sample_loss.detach()

    # 【修改点 3】: 从损失映射到置信度真值
    # temperature 是一个重要的超参数，用于调节映射的敏感度
    temperature = 0.1 # 建议从config传入
    
    # per_sample_loss_detached 值越小，confidence_gt 越接近1.0
    # confidence_gt 形状为 (B, N_ref-1)
    confidence_gt = torch.exp(-per_sample_loss_detached[:, 1:] / temperature)
    
    # 假设 pred_confs 的形状是 (B, N, 1)，其中第0个是查询图的
    # 提取参考图的logits
    pred_confs_ref_logits = pred_confs[:, 1:, 0] # 形状 (B, N-1)
    
    # 对logits使用sigmoid得到0-1之间的分数
    pred_scores = torch.sigmoid(pred_confs_ref_logits)
    
    # 使用 L1 Loss 计算回归损失
    confidence_loss = F.l1_loss(pred_scores, confidence_gt)
    loss_dict['loss_conf'] = confidence_loss.detach()

    # --- 总损失 ---
    total_loss = loss_pv_map * config.train.loss_weights.weight_pv_map + \
                 confidence_loss * config.train.loss_weights.weight_conf
                 
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