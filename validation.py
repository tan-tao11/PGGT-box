import torch
import cv2
import numpy as np

from src.data.onepose import Onepose
from tqdm import tqdm

from utils.geometry import compute_pose_via_pnp
from src.metrics.pose_metric import rotation_matrix_angle_error
from src.losses.loss import cal_loss
from src.losses.loss_with_conf import cal_loss_with_confidence, compute_camera_loss
from utils.math_utils import sigmoid

def compute_metrics(pred_bbox, pred_conf, query_info, ref_info, config, is_train=True):
    query_pose = query_info['query_pose'].numpy()
    query_intrinsics = query_info['query_intrinsics'].numpy()
    gt_query_bbox = query_info['bbox_2d'].numpy()
    ref_bbox = ref_info['bbox_2d'].numpy()
    bbox_3d = ref_info['bbox_3d'].numpy()

    # Predicted rotation residual
    # all_pred_poses = pred[0].detach().to(torch.float32).cpu().numpy()  #  (B, N, 9)
    # all_pred_conf_logits = pred[1][:, 1:].detach().to(torch.float32).cpu().numpy()  #  (B, N, 1)
    pred_bbox = pred_bbox[:, 1:].detach().to(torch.float32).cpu().numpy()  # Ignore the first pose (B, N-1, 16)
    pred_conf = pred_conf[:, 1:, 0].detach().to(torch.float32).cpu().numpy()  # Ignore the first confidence (B, N-1, 1)
    residual_bbox_pred = pred_bbox
    # Select the residual with highest confidence
    pred_conf = sigmoid(pred_conf) 
    best_pred_indices = np.argmax(pred_conf, axis=1)

    B = ref_bbox.shape[0]
    residual_bbox_pred_best = residual_bbox_pred[np.arange(B), best_pred_indices].reshape(-1, 8, 2)
    # Ref rotation matrices
    ref_bbox_best = ref_bbox[np.arange(B), best_pred_indices]
    # Compute the query bbox
    query_bbox_pred = (ref_bbox_best - residual_bbox_pred_best) * config.data.target_size
    # query_bbox_pred = query_info['bbox_2d'].numpy()* config.data.target_size
    # Compute query pos via pnp
    query_rot_pred = np.zeros((B, 3, 3))
    query_trans_pred = np.zeros((B, 3))
    for i in range(B):
        query_rot_pred[i], query_trans_pred[i] = compute_pose_via_pnp(query_bbox_pred[i], bbox_3d[i], query_intrinsics[i][0])
    
    # Compute metrics
    rotation_angle_error = rotation_matrix_angle_error(query_rot_pred, query_pose[:, 0, :3, :3])
    translation_error = np.linalg.norm(query_trans_pred - query_pose[:, 0, :3, 3]) * 1000

    return np.mean(rotation_angle_error), np.mean(translation_error), query_bbox_pred[0], gt_query_bbox[0, 0], ref_bbox_best[0], best_pred_indices[0]
    

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
        "loss_bbox": 0,
        "loss_conf": 0,
    }
    len_data = len(data_loader)
    vis_itr = len_data // 5
    with torch.no_grad():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        for idx, data in tqdm(enumerate(data_loader)):

            with torch.cuda.amp.autocast(dtype=dtype):
                # Forward pass
                outputs = model(data, device)

            # Calculate loss
            _, loss_dict = compute_camera_loss(outputs[0], outputs[1], data, config=config)
            
            for key, value in loss_dict.items():
                loss_dict_all[key] += value.item() / len_data

            # Compute metrics
            query_info = data['query_info']
            ref_info = data['ref_info']
            rotation_angle_error, translation_error, query_bbox, gt_query_bbox, ref_bbox, ref_indice = compute_metrics(outputs[0][-1], outputs[1][-1], query_info, ref_info, config, is_train=False)
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
                反归一化图像张量，输入 tensor 为 (3, H, W)
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

