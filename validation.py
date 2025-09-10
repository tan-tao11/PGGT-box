import torch
import cv2
import numpy as np
import time

from src.data.onepose import Onepose
from tqdm import tqdm

from utils.geometry import compute_pose_via_pnp
from src.metrics.pose_metric import rotation_matrix_angle_error
from src.losses.loss import cal_loss
from src.losses.loss_with_conf import cal_loss_with_confidence, compute_camera_loss
from utils.math_utils import sigmoid
from src.utils.pixel_voting import find_keypoints_from_vector_map, _find_single_keypoint_ransac_refined

def compute_metrics(pred_pv_map_offset, pred_confs, query_info, ref_info, config, is_train=True):
    query_pose = query_info['query_pose'].numpy()
    query_intrinsics = query_info['query_intrinsics'].numpy()
    gt_query_bbox = query_info['bbox_2d'].numpy()
    ref_bbox = ref_info['bbox_2d'].numpy()
    bbox_3d = ref_info['bbox_3d'].numpy()
    ref_pv_maps = ref_info['pv_maps'].numpy()

    B = pred_pv_map_offset.shape[0]

    # 选择置信度最高的参考视图
    # pred_confs = torch.sigmoid(pred_confs[:, 1:].to(torch.float32)).cpu().numpy()
    best_pred_indices = np.argmax(pred_confs[:, 1:].to(torch.float32).cpu().numpy(), axis=1)
    ref_pv_maps_best = ref_pv_maps[np.arange(ref_pv_maps.shape[0]), best_pred_indices[:,0]]
    pred_pv_map_offset_best = pred_pv_map_offset[np.arange(pred_pv_map_offset.shape[0]), best_pred_indices[:,0]+1].to(torch.float32).cpu().numpy()
    ref_bbox_best = ref_bbox[np.arange(ref_bbox.shape[0]), best_pred_indices[:,0]]   
    # 计算查询图像的向量图
    query_pv_maps = pred_pv_map_offset_best + ref_pv_maps_best
    # 计算关键点坐标，RANSAC
    query_rot_pred = np.zeros((B, 3, 3))
    query_trans_pred = np.zeros((B, 3))
    keypoints_coords_list = []
    for i in range(B):
        keypoints_coords = find_keypoints_from_vector_map(query_pv_maps[i])
        # keypoints_coords = _find_single_keypoint_ransac_refined(query_pv_maps[i])
        keypoints_coords_list.append(keypoints_coords)
        query_rot_pred[i], query_trans_pred[i] = compute_pose_via_pnp(keypoints_coords, bbox_3d[i], query_intrinsics[i][0])
    
    # Compute metrics
    rotation_angle_error = rotation_matrix_angle_error(query_rot_pred, query_pose[:, 0, :3, :3])
    translation_error = np.linalg.norm(query_trans_pred - query_pose[:, 0, :3, 3]) * 1000

    return np.mean(rotation_angle_error), np.mean(translation_error), keypoints_coords_list[0], gt_query_bbox[0, 0], ref_bbox_best[0], best_pred_indices[0, 0]
    

def validate_model(model, config, device, writer=None):
    # model.eval()

    # Load dataset
    dataset = Onepose(config.val)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    losses_trans = []
    losses_rot = []
    rotation_angle_errors = []
    translation_errors = []
    vis_images = []
    print("Starting validation...")
    loss_dict_all = {
        "total_loss": 0,
        "loss_pv_map": 0,
        "loss_conf": 0,
    }
    loss_conf = {}
    len_data = len(data_loader)
    vis_itr = len_data // 5
    with torch.no_grad():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        for idx, data in tqdm(enumerate(data_loader)):
            with torch.cuda.amp.autocast(dtype=dtype):
                # Forward pass
                pred_pv_map_offset, pred_conf = model(data, device)
            # Calculate loss
            _, loss_dict = cal_loss(pred_pv_map_offset, pred_conf, data, config=config)
            
            for key, value in loss_dict.items():
                loss_dict_all[key] += value.item() / len_data
            loss_conf[idx] = loss_dict['loss_conf'].item()
            # Compute metrics
            query_info = data['query_info']
            ref_info = data['ref_info']
            # gt_pv_map_offset = data['pv_maps_offset'].to(pred_pv_map_offset.device)
            rotation_angle_error, translation_error, query_bbox, gt_query_bbox, ref_bbox, ref_indice = compute_metrics(pred_pv_map_offset, pred_conf, query_info, ref_info, config, is_train=False)
            if idx % vis_itr == 0:
                vis_image = vis(data['images'][0, 0], query_bbox, gt_query_bbox, config)
                vis_images.append(vis_image)
                vis_image = vis(data['images'][0, ref_indice+1], ref_bbox, ref_bbox, config)
                vis_images.append(vis_image)
            # gt_pose_encodings = data['pose_residual_gt'].to(device)
            # rotation_angle_error, translation_error = compute_metrics(gt_pose_encodings, outputs[1][-1], query_info, ref_info, config)
            rotation_angle_errors.append(rotation_angle_error)
            translation_errors.append(translation_error)
    
    # Compute cm-degree metric
    ratio = sum((a < 5 and b < 50) for a, b in zip(rotation_angle_errors, translation_errors)) / len(rotation_angle_errors)

    return loss_dict_all, sum(rotation_angle_errors) / len(rotation_angle_errors), sum(translation_errors) / len(translation_errors), ratio, vis_images

def vis(image, query_bbox, gt_query_bbox, config):
    def denormalize_img(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                """
                反归一化图像张量，输入 tensor 为 (3, H, W)·
                """
                mean = np.array(mean).reshape(3, 1, 1)
                std = np.array(std).reshape(3, 1, 1)
                return ((img * std + mean)*255).astype(np.uint8)
    image = denormalize_img(image.numpy())
    image = image.transpose(1, 2, 0)
    image = np.ascontiguousarray(image)
    for bbox in query_bbox:
        if abs(bbox[0]) < 1.0:
             cv2.circle(image, (int(bbox[0]*config.data.target_size), int(bbox[1]*config.data.target_size)), 5, (0, 255, 0), 2)
        else:
            cv2.circle(image, (int(bbox[0]), int(bbox[1])), 5, (0, 255, 0), 2)
    for bbox in gt_query_bbox:
        if abs(bbox[0]) < 1.0:
             cv2.circle(image, (int(bbox[0]*config.data.target_size), int(bbox[1]*config.data.target_size)), 5, (255, 0, 0), 2)
        else:
            cv2.circle(image, (int(bbox[0]), int(bbox[1])), 5, (0, 255, 0), 2)
    
    return image.transpose(2, 0, 1)

