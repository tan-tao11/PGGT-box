import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def project_points_cam(points, intrinsics):
    """
    Projects 3D points to 2D image coordinates.

    Args:
        points (np.ndarray): 3D points of shape (N, 3)
        intrinsics (np.ndarray): Camera intrinsic matrix of shape (3, 3)

    Returns:
        np.ndarray: 2D image coordinates of shape (N, 2)
        np.ndarray: depth values of shape (N, 1)
    """
    projected = np.dot(points, intrinsics.T)
    return projected[:, :2] / projected[:, 2:], projected[:, 2:]

def project_points_obj(points, extrinsics, intrinsics):
    """
    Projects 3D points to 2D image coordinates.

    Args:
        points (np.ndarray): 3D points of shape (N, 3)
        extrinsics (np.ndarray): Camera extrinsic matrix of shape (3, 4)
        intrinsics (np.ndarray): Camera intrinsic matrix of shape (3, 3)

    Returns:
        np.ndarray: 2D image coordinates of shape (N, 2)
        np.ndarray: depth values of shape (N, 1)
    """
    points = np.dot(points, extrinsics.T)
    projected = np.dot(points, intrinsics.T)
    return projected[:, :2] / projected[:, 2:], projected[:, 2:]

def project_object_centers(t_all, K_all):
    """
    Projects the center of N objects to image plane using corresponding camera intrinsics and object poses.

    Args:
        t_all (np.ndarray): Translation vectors of shape (N, 3).
        K_all (np.ndarray): Camera intrinsic matrices of shape (N, 3, 3).

    Returns:
        uv_all (np.ndarray): 2D image coordinates of shape (N, 2), where each row is (u, v).
    """

    N = t_all.shape[0]
    R_all = np.eye(3).reshape(1, 3, 3).repeat(N, axis=0)  # Identity rotation for each object

    # Construct [R | t] for each object → shape (N, 3, 4)
    Rt_all = np.concatenate([R_all, t_all[:, :, np.newaxis]], axis=2)

    # Define homogeneous object center (0, 0, 0, 1) for each object → shape (N, 4, 1)
    obj_center = np.tile(np.array([[0, 0, 0, 1]]), (N, 1)).reshape(N, 4, 1)

    # Transform object center to camera coordinate frame → shape (N, 3, 1)
    X_cam_all = np.matmul(Rt_all, obj_center)

    # Project into image plane: x_img = K @ X_cam → shape (N, 3, 1)
    x_img_all = np.matmul(K_all, X_cam_all)

    # Remove last dimension → shape (N, 3)
    x_img_all = x_img_all.squeeze(-1)

    # Normalize by depth (z) → shape (N, 2)
    u = x_img_all[:, 0] / x_img_all[:, 2]
    v = x_img_all[:, 1] / x_img_all[:, 2]
    uv_all = np.stack([u, v], axis=1)
    z_all = x_img_all[:, 2:3]  # Depth values

    return uv_all, z_all[:, 0]

def to_SITE(o_xy, depth, bb_centers, zoom_ratios, crop_size):
    """
    Converts translation vector to SITE format.

    Args:
        o_xy (np.ndarray): Object center in image coordinates of shape (N, 2).
        depth (np.ndarray): Depth values of shape (N).
        bb_centers (np.ndarray): Bounding box centers of shape (N, 2).
        zoom_ratios (np.ndarray): Zoom ratios of shape (N).
        crop_size (np.ndarray): Cropped image size.

    Returns:
        np.ndarray: SITE representation of the translation of shape (N, 3).
    """
    delta_x = (o_xy[:, 0] - bb_centers[:, 0]) / crop_size[:, 0]
    delta_y = (o_xy[:, 1] - bb_centers[:, 1]) / crop_size[:, 1]
    delta_z = depth / zoom_ratios

    return np.stack([delta_x, delta_y, delta_z], axis=1)

def translation_parameterization(trans, intrinsics, bb_centers, zoom_ratios, img_size):
    """
    Parameterizes the object pose into 6D rotation and SITE translation.

    Args:
        trans (np.ndarray): Object translation of shape (N, 3).
        intrinsics (np.ndarray): Camera intrinsic matrix of shape (N, 3, 3).
        bb_centers (np.ndarray): Bounding box centers of shape (N, 2).
        zoom_ratios (np.ndarray): Zoom ratios of shape (N).

    Returns:
        np.ndarray: 6D rotation of shape (N, 6).
        np.ndarray: SITE translation of shape (N, 3).
    """
    proj_centers, proj_depths = project_object_centers(
            t_all = trans,
            K_all = intrinsics
        )

    # Normalize projected object centers
    proj_centers = proj_centers / img_size
    # trans_param = to_SITE(
    #     pro_centers, 
    #     proj_depths,
    #     bb_centers,
    #     zoom_ratios,
    #     img_size
    # )
    trans_param = np.concatenate([proj_centers, proj_depths[:, np.newaxis]], axis=1)

    return trans_param

def translation_from_center_depth(trans_param, intrinsics):
    """
    Converts center-depth params to 3D translation vector.

    Args:
        trans_param (np.ndarray): Center-depth parameters of shape (N, 2).
        intrinsics (np.ndarray): Camera intrinsics of shape (N, 3, 3).

    Returns:
        np.ndarray: 3D translation vectors of shape (N, 3).
    """
    o_x, o_y, depth = trans_param[:, 0], trans_param[:, 1], trans_param[:, 2]

    # Recover 3D translation
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    t_x = (o_x - cx) * depth / fx
    t_y = (o_y - cy) * depth / fy
    t_z = depth

    return np.stack([t_x, t_y, t_z], axis=1)

def translation_from_SITE(trans_param, intrinsics, bb_centers, zoom_ratios, img_size):
    """
    Converts SITE translation back to 3D translation vector.

    Args:
        trans_param (np.ndarray): SITE translation of shape (N, 3).
        intrinsics (np.ndarray): Camera intrinsics of shape (N, 3, 3).
        bb_centers (np.ndarray): Bounding box centers of shape (N, 2).
        zoom_ratios (np.ndarray): Zoom ratios of shape (N,).
        img_size (np.ndarray): Image sizes of shape (N, 2).

    Returns:
        np.ndarray: 3D translation vectors of shape (N, 3).
    """
    delta_x, delta_y, delta_z = trans_param[:, 0], trans_param[:, 1], trans_param[:, 2]

    # Recover depth
    depths = delta_z * zoom_ratios  # (N,)

    # Recover image coordinates of object center o_xy
    o_x = delta_x * img_size[:, 0] + bb_centers[:, 0]
    o_y = delta_y * img_size[:, 1] + bb_centers[:, 1]
    o_xy = np.stack([o_x, o_y], axis=1)  # (N, 2)

    # Recover 3D translation
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    t_x = (o_xy[:, 0] - cx) * depths / fx
    t_y = (o_xy[:, 1] - cy) * depths / fy
    t_z = depths

    return np.stack([t_x, t_y, t_z], axis=1)  # (N, 3)

def normalize(v, axis=-1, eps=1e-8):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + eps)

def matrix_to_rot6d_numpy(rot_mats):
    """
    将旋转矩阵转换为 6D 旋转表示（取前两列拼接）

    参数:
    - rot_mats: ndarray, shape 为 (3, 3) 或 (N, 3, 3)

    返回:
    - rot6d: ndarray, shape 为 (6,) 或 (N, 6)
    """
    rot_mats = np.asarray(rot_mats)

    if rot_mats.ndim == 2 and rot_mats.shape == (3, 3):
        # 单个矩阵输入
        r1 = rot_mats[:, 0]
        r2 = rot_mats[:, 1]
        rot6d = np.concatenate([r1, r2], axis=0)  # shape: (6,)
    elif rot_mats.ndim == 3 and rot_mats.shape[1:] == (3, 3):
        # 批量输入
        r1 = rot_mats[:, :, 0]  # shape: (N, 3)
        r2 = rot_mats[:, :, 1]  # shape: (N, 3)
        rot6d = np.concatenate([r1, r2], axis=1)  # shape: (N, 6)
    else:
        raise ValueError("Input must be of shape (3,3) or (N,3,3)")

    return rot6d

# def rot6d_to_matrix(x):
#     """
#     将 6D 旋转表示转换为 3x3 旋转矩阵（NumPy 版本）

#     参数:
#     - x: ndarray of shape (N, 6)

#     返回:
#     - rot_mat: ndarray of shape (N, 3, 3)
#     """
#     assert x.shape[-1] == 6, "Input must have shape (N, 6)"

#     a1 = normalize(x[:, 0:3], axis=1)   # shape: (N, 3)
#     a2 = x[:, 3:6]                      # shape: (N, 3)

#     # Gram-Schmidt 正交化
#     b = a2 - np.sum(a1 * a2, axis=1, keepdims=True) * a1
#     b = normalize(b, axis=1)
#     c = np.cross(a1, b)

#     # 拼接为旋转矩阵 (N, 3, 3)
#     rot_mat = np.stack([a1, b, c], axis=-1)  # shape: (N, 3, 3)
#     return rot_mat

def rot6d_to_matrix(x):
    """
    将 6D 旋转表示转换为 3x3 旋转矩阵（NumPy 版本）

    参数:
    - x: ndarray of shape (..., 6)  # 支持 (6,)、(N, 6)、(B, N, 6) 等

    返回:
    - rot_mat: ndarray of shape (..., 3, 3)
    """
    assert x.shape[-1] == 6, f"Input last dimension must be 6, got {x.shape[-1]}"

    # 保存原始形状以便最后reshape
    original_shape = x.shape[:-1]
    x = x.reshape(-1, 6)  # 展平所有前面的维度

    a1 = normalize(x[:, 0:3], axis=1)   # shape: (M, 3), M = prod(original_shape)
    a2 = x[:, 3:6]                      # shape: (M, 3)

    # Gram-Schmidt 正交化
    b = a2 - np.sum(a1 * a2, axis=1, keepdims=True) * a1
    b = normalize(b, axis=1)
    c = np.cross(a1, b)

    # 拼接为旋转矩阵 (M, 3, 3)
    rot_mat = np.stack([a1, b, c], axis=-1)  # shape: (M, 3, 3)

    # 恢复原始形状
    rot_mat = rot_mat.reshape(*original_shape, 3, 3)
    return rot_mat

def select_min_rotation_angle_matrix(rot_mats: np.ndarray) -> np.ndarray:
    """
    从一组旋转矩阵中选出旋转角度最小的矩阵。
    支持两种输入形状：
    1. (N, 3, 3) - 返回单个最小矩阵
    2. (B, N, 3, 3) - 对每个batch返回对应的最小矩阵

    Args:
        rot_mats (np.ndarray): 形状为 (N, 3, 3) 或 (B, N, 3, 3) 的旋转矩阵数组。

    Returns:
        np.ndarray: 形状为 (3, 3) 或 (B, 3, 3) 的旋转角度最小的旋转矩阵。
        int or np.ndarray: 最小角度对应的索引或索引数组
    """
    # 检查输入维度
    if rot_mats.ndim == 3:
        # 原始情况 (N, 3, 3)
        traces = np.trace(rot_mats, axis1=1, axis2=2)  # (N,)
        cos_theta = (traces - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.arccos(cos_theta)  # (N,)
        
        min_idx = np.argmin(angles)
        return rot_mats[min_idx], min_idx
    elif rot_mats.ndim == 4:
        # batch处理 (B, N, 3, 3)
        traces = np.trace(rot_mats, axis1=2, axis2=3)  # (B, N)
        cos_theta = (traces - 1) / 2
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.arccos(cos_theta)  # (B, N)
        
        min_indices = np.argmin(angles, axis=1)  # (B,)
        
        # 使用高级索引获取每个batch的最小矩阵
        batch_indices = np.arange(rot_mats.shape[0])
        min_mats = rot_mats[batch_indices, min_indices]
        
        return min_mats, min_indices
    else:
        raise ValueError(f"输入维度必须是3或4，但得到了{rot_mats.ndim}")
    
def select_min_translation(translations):
    """
    找出每一批中模长最小的平移向量的序号（NumPy 版本）
    
    参数:
        arr (np.ndarray): 形状为 (B, N, 3) 的数组，B 是 batch 大小，N 是向量数量，3 是平移向量
    
    返回:
        np.ndarray: 形状为 (B,) 的数组，包含每一批中最小模长向量的索引
    """
    # 计算每个向量的模长的平方（避免开平方以节省计算）
    norms_sq = np.sum(translations ** 2, axis=2)  # 形状 (B, N)
    
    # 找出每一批中最小模长的索引
    min_indices = np.argmin(norms_sq, axis=1)  # 形状 (B,)
    
    return min_indices

# def select_min_rotation_angle_matrix(rot_mats: np.ndarray) -> np.ndarray:
#     """
#     从一组旋转矩阵中选出旋转角度最小的矩阵。

#     Args:
#         rot_mats (np.ndarray): 形状为 (N, 3, 3) 的旋转矩阵数组。

#     Returns:
#         np.ndarray: 形状为 (3, 3) 的旋转角度最小的旋转矩阵。
#     """
#     # 计算 trace(R) 对应的旋转角度
#     traces = np.trace(rot_mats, axis1=1, axis2=2)  # (N,)
#     cos_theta = (traces - 1) / 2
#     # clip 是为了防止 arccos 数值误差越界
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     angles = np.arccos(cos_theta)  # (N,)

#     # 找到最小角度对应的索引
#     min_idx = np.argmin(angles)
#     return rot_mats[min_idx], min_idx

def compute_normalization_scale(bbox_vertices, mode="max_extent"):
    """
    计算物体大小归一化系数。

    参数:
        bbox_vertices: (8, 3) ndarray，包围盒8个角点的坐标
        mode: str，归一化方式，支持：
            - "max_extent": 按最长边归一化到1（常见）
            - "unit_sphere": 归一化到单位球（center到最远点为1）
            - "diag": 按包围盒对角线长度归一化

    返回:
        scale: float，归一化系数（即应除以的值）
    """
    bbox_vertices = np.asarray(bbox_vertices)
    assert bbox_vertices.shape == (8, 3), "bbox_vertices 应为 (8, 3) 数组"

    if mode == "max_extent":
        min_pt = bbox_vertices.min(axis=0)
        max_pt = bbox_vertices.max(axis=0)
        extent = max_pt - min_pt
        scale = np.max(extent)

    elif mode == "unit_sphere":
        center = bbox_vertices.mean(axis=0)
        dists = np.linalg.norm(bbox_vertices - center, axis=1)
        scale = np.max(dists)

    elif mode == "diag":
        min_pt = bbox_vertices.min(axis=0)
        max_pt = bbox_vertices.max(axis=0)
        scale = np.linalg.norm(max_pt - min_pt)

    else:
        raise ValueError(f"未知归一化方式: {mode}")

    return scale

def warp_intrinsics(intrinsics, crop_centers, crop_shapes, zoom_ratios):
    """
    args:
        intrinsics: (N, 3, 3)
        crop_centers: (N, 2)
        crop_shapes: (N, 2)
        zoom_ratios: (N, 2)
    对相机内参进行裁剪和缩放。
    """
    
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]

    # 计算裁剪后的内参
    cx_crop = cx - (crop_centers[:, 0] - crop_shapes[:, 0] / 2) 
    cy_crop = cy - (crop_centers[:, 1] - crop_shapes[:, 1] / 2)
    
    # 计算缩放后的内参
    fx_zoom = fx * zoom_ratios[:, 0]
    fy_zoom = fy * zoom_ratios[:, 1]
    cx_zoom = cx_crop * zoom_ratios[:, 0]
    cy_zoom = cy_crop * zoom_ratios[:, 1]

    intrinsics_new = np.zeros_like(intrinsics)
    intrinsics_new[:, 0, 0] = fx_zoom
    intrinsics_new[:, 1, 1] = fy_zoom
    intrinsics_new[:, 0, 2] = cx_zoom
    intrinsics_new[:, 1, 2] = cy_zoom
    intrinsics_new[:, 2, 2] = 1

    return intrinsics_new

def project_3d_box_to_2d(bbox_3d, poses, intrinsics):
    """
    将一个 (8,3) 的3D包围框投影到 N 个 2D 相机平面上 (numpy 版本)

    Args:
        bbox_3d: (8,3) 3D 包围框的角点 (x,y,z)
        poses: (N,4,4) 相机位姿矩阵（世界到相机变换）
        intrinsics: (N,3,3) 相机内参

    Returns:
        projected_2d: (N,8,2) 投影到每个相机的 2D 坐标
    """
    # (8,3) -> (8,4)，加上齐次坐标 1
    bbox_homo = np.concatenate([bbox_3d, np.ones((bbox_3d.shape[0], 1))], axis=-1)  # (8,4)

    N = poses.shape[0]
    projected_points = []

    for i in range(N):
        # 世界坐标 -> 相机坐标
        cam_points = (poses[i] @ bbox_homo.T).T  # (8,4)

        # 丢弃齐次最后一列 (x,y,z)
        cam_points = cam_points[:, :3]

        # 相机坐标 -> 像素坐标
        pixels_homo = (intrinsics[i] @ cam_points.T).T  # (8,3)

        # 齐次除法
        pixels_2d = pixels_homo[:, :2] / pixels_homo[:, 2:3]  # (8,2)

        projected_points.append(pixels_2d)

    return np.stack(projected_points, axis=0)  # (N,8,2)

def compute_pose_via_pnp(pts_2d, pts_3d, camera_matrix, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1))
    
    # Ransac PnP
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d,
        pts_2d,
        camera_matrix,
        dist_coeffs,
        reprojectionError=5.0,   # 投影误差阈值（像素）
        iterationsCount=100,     # RANSAC迭代次数
        confidence=0.99,         # 置信度
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        rot_mat, _ = cv2.Rodrigues(rvec)  # 旋转矩阵
        # pose = np.zeros((4, 4), dtype=np.float32)
        # pose[:3, :3] = rot_mat
        # pose[:3, 3] = tvec.flatten()
        # pose[3, 3] = 1
        return rot_mat, tvec.flatten()
    else:
        return np.eye(3), np.zeros(3)