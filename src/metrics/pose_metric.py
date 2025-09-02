import numpy as np

# def rotation_matrix_angle_error(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
#     """
#     批量计算两个 (N, 3, 3) 旋转矩阵之间的旋转角度误差（弧度）。

#     Args:
#         R1 (np.ndarray): 形状为 (N, 3, 3) 的第一个旋转矩阵数组。
#         R2 (np.ndarray): 形状为 (N, 3, 3) 的第二个旋转矩阵数组。

#     Returns:
#         np.ndarray: 形状为 (N,) 的角度误差数组（单位：弧度）。
#     """
#     assert R1.shape == R2.shape and R1.shape[1:] == (3, 3)

#     # 相对旋转矩阵：R_rel = R1^T @ R2
#     R_rel = np.matmul(R1.transpose(0, 2, 1), R2)

#     # trace(R_rel)
#     traces = np.trace(R_rel, axis1=1, axis2=2)

#     # 角度公式，clip 防止浮点精度超出 [-1, 1]
#     cos_theta = (traces - 1) / 2
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     angles = np.arccos(cos_theta)
#     angles_deg = np.degrees(angles)

#     return angles_deg

def rotation_matrix_angle_error(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """
    计算两个旋转矩阵之间的旋转角度误差（弧度），支持批量处理。

    参数:
        R1 (np.ndarray): 形状为 (..., 3, 3) 的第一个旋转矩阵数组
        R2 (np.ndarray): 形状为 (..., 3, 3) 的第二个旋转矩阵数组
        (支持形状如 (3,3)、(N,3,3)、(B,N,3,3) 等)

    返回:
        np.ndarray: 形状为 (...) 的角度误差数组（单位：度）
    """
    assert R1.shape == R2.shape and R1.shape[-2:] == (3, 3), \
        f"输入形状不匹配或不是旋转矩阵: R1 {R1.shape}, R2 {R2.shape}"

    # 保存原始形状以便最后reshape
    original_shape = R1.shape[:-2]
    R1 = R1.reshape(-1, 3, 3)  # (M, 3, 3)
    R2 = R2.reshape(-1, 3, 3)  # (M, 3, 3)

    # 相对旋转矩阵：R_rel = R1^T @ R2
    R_rel = np.matmul(np.transpose(R1, (0, 2, 1)), R2)  # (M, 3, 3)

    # trace(R_rel)
    traces = np.trace(R_rel, axis1=1, axis2=2)  # (M,)

    # 角度计算
    cos_theta = (traces - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles_rad = np.arccos(cos_theta)
    angles_deg = np.degrees(angles_rad)  # (M,)

    # 恢复原始形状
    return angles_deg.reshape(original_shape)