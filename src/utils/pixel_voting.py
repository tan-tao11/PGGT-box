import numpy as np
import multiprocessing
from functools import partial
import math
# 推荐安装numba库: pip install numba
from numba import jit, njit, prange

@njit(parallel=True)
def generate_pvmap_single_numba(keypoints_2d, H, W):
    """
    使用 Numba 加速的单图像 PVMap 生成
    keypoints_2d: (K, 2)
    返回: (H, W, K*2)
    """
    K = keypoints_2d.shape[0]
    gt_vector_field = np.zeros((H, W, K*2), dtype=np.float32)

    for i in prange(K):   # 并行关键点
        kpt_x, kpt_y = keypoints_2d[i]
        for y in range(H):
            for x in range(W):
                dx = kpt_x - x
                dy = kpt_y - y
                norm = (dx*dx + dy*dy) ** 0.5 + 1e-8
                gt_vector_field[y, x, i*2]   = dx / norm
                gt_vector_field[y, x, i*2+1] = dy / norm
    return gt_vector_field


def generate_pvmap_batch_numba(all_keypoints_2d, output_shape):
    """
    批量版本，逐张调用 numba 编译的单图函数。
    all_keypoints_2d: (N, K, 2)
    output_shape: (H, W)
    返回: (N, H, W, K*2)
    """
    N, K, _ = all_keypoints_2d.shape
    H, W = output_shape
    pv_maps = np.zeros((N, H, W, K*2), dtype=np.float32)

    for n in range(N):
        pv_maps[n] = generate_pvmap_single_numba(all_keypoints_2d[n], H, W)
    
    return pv_maps

def generate_pvmap_batch(all_keypoints_2d, output_shape):
    """
    高效地批量生成PVNet向量场真值。
    此函数完全向量化，没有Python循环。

    参数:
    all_keypoints_2d (np.array): 所有图像的关键点坐标。
                                形状: (N, K, 2)，N是图像数, K是关键点数。
    output_shape (tuple): (H, W)，输出向量图的尺寸。

    返回:
    np.array: 形状为 (N, H, W, K*2) 的向量场真值数组。
    """
    N, K, _ = all_keypoints_2d.shape
    H, W = output_shape

    # 1. 创建坐标网格（只需一次）
    # y_coords, x_coords 形状均为 (H, W)
    y_coords, x_coords = np.indices((H, W), dtype=np.float32)

    # 2. 准备广播
    # 将关键点坐标和像素坐标的维度进行扩展，以便它们可以相互广播
    
    # kpts_x 的形状: (N, K) -> (N, K, 1, 1)
    kpts_x = all_keypoints_2d[:, :, 0][:, :, np.newaxis, np.newaxis]
    # kpts_y 的形状: (N, K) -> (N, K, 1, 1)
    kpts_y = all_keypoints_2d[:, :, 1][:, :, np.newaxis, np.newaxis]
    
    # x_coords 的形状: (H, W) -> (1, 1, H, W)
    # y_coords 的形状: (H, W) -> (1, 1, H, W)
    # NumPy会自动处理 (H, W) 到 (1, 1, H, W) 的广播，但为了清晰，可以手动扩展
    
    # 3. 一次性计算所有原始向量（利用广播）
    # raw_vx 和 raw_vy 的形状将是 (N, K, H, W)
    raw_vx = kpts_x - x_coords
    raw_vy = kpts_y - y_coords

    # 4. 一次性计算所有模长并进行归一化
    norm = np.sqrt(raw_vx**2 + raw_vy**2) + 1e-8
    unit_vx = raw_vx / norm
    unit_vy = raw_vy / norm
    # unit_vx, unit_vy 的形状仍为 (N, K, H, W)

    # 5. 将 vx 和 vy 堆叠并重塑为最终格式
    # 堆叠成 (N, K, H, W, 2)
    stacked_vectors = np.stack([unit_vx, unit_vy], axis=-1)
    
    # 调整维度顺序为 (N, H, W, K, 2)
    transposed_vectors = stacked_vectors.transpose(0, 2, 3, 1, 4)
    
    # 最后，将最后两个维度 (K, 2) 合并为 K*2，得到最终输出
    # 最终形状: (N, H, W, K*2)
    gt_vector_fields = transposed_vectors.reshape(N, H, W, K * 2)
    
    return gt_vector_fields

# def generate_pvmap_wo_mask(keypoints_2d, output_shape):
#     """
#     高效地、无掩码地批量生成PVNet向量场真值。
#     此函数完全向量化，以替代多层嵌套的Python for循环。

#     参数:
#     keypoints_2d (np.array): 关键点坐标，形状为 (K, 2)，K是关键点数量。
#     output_shape (tuple): (H, W)，输出向量图的尺寸。

#     返回:
#     np.array: 形状为 (H, W, K*2) 的密集向量图真值数组。
#     """
#     H, W = output_shape
#     K = keypoints_2d.shape[0]

#     # 1. 创建坐标网格（只需一次）
#     # y_coords, x_coords 形状均为 (H, W)
#     y_coords, x_coords = np.indices((H, W), dtype=np.float32)

#     # 2. 准备广播 (核心技巧)
#     # 将关键点坐标的维度进行扩展，以便它们可以与整个像素坐标网格进行运算。
#     # kpts_x 的形状: (K,) -> (K, 1, 1)
#     kpts_x = keypoints_2d[:, 0][:, np.newaxis, np.newaxis]
#     # kpts_y 的形状: (K,) -> (K, 1, 1)
#     kpts_y = keypoints_2d[:, 1][:, np.newaxis, np.newaxis]
    
#     # 3. 一次性计算所有原始向量
#     # (K, 1, 1) - (H, W) -> 广播成 (K, H, W)
#     raw_vx = kpts_x - x_coords
#     raw_vy = kpts_y - y_coords

#     # 4. 一次性计算所有模长并进行归一化
#     # 所有操作都是在 (K, H, W) 的张量上进行的，非常高效
#     norm = np.sqrt(raw_vx**2 + raw_vy**2) + 1e-8
#     unit_vx = raw_vx / norm
#     unit_vy = raw_vy / norm

#     # 5. 将 vx 和 vy 堆叠并重塑为最终的 (H, W, K*2) 格式
#     # a. 堆叠成 (K, H, W, 2)，最后一个维度代表 (vx, vy)
#     stacked_vectors = np.stack([unit_vx, unit_vy], axis=-1)
    
#     # b. 调整维度顺序为 (H, W, K, 2)
#     transposed_vectors = stacked_vectors.transpose(1, 2, 0, 3)
    
#     # c. 最后，将最后两个维度 (K, 2) 合并为 K*2
#     gt_vector_field = transposed_vectors.reshape(H, W, K * 2)
    
#     return gt_vector_field

import numpy as np
from numba import njit

@njit(cache=True) # 我们去掉了 parallel=True
def generate_all_pvmaps_numba_serial(all_keypoints_2d, H, W):
    """
    使用Numba JIT编译整个串行循环，以实现最高单核性能和最低内存占用。
    
    参数:
    all_keypoints_2d (np.array): 形状 (N, K, 2)
    H (int): 高度
    W (int): 宽度
    """
    N, K, _ = all_keypoints_2d.shape
    
    # 最终的输出数组
    all_gt_vector_fields = np.zeros((N, H, W, K * 2), dtype=np.float32)

    # Numba会把下面这个四层嵌套的循环编译成一个整体
    for i in range(N):
        keypoints_2d = all_keypoints_2d[i]
        for k in range(K):
            kpt_x, kpt_y = keypoints_2d[k]
            for y in range(H):
                for x in range(W):
                    dx = kpt_x - x
                    dy = kpt_y - y
                    norm = (dx**2 + dy**2)**0.5 + 1e-8
                    all_gt_vector_fields[i, y, x, k*2]   = dx / norm
                    all_gt_vector_fields[i, y, x, k*2+1] = dy / norm
                    
    return all_gt_vector_fields

@njit(parallel=True)
def generate_pvmap_wo_mask(keypoints_2d, output_shape):
    """
    生成PVNet向量场真值，但不使用掩码。
    （警告：这仅用于实验，不推荐用于正式训练）
    """
    H, W = output_shape
    K = keypoints_2d.shape[0]
    gt_vector_field = np.zeros((H, W, K*2), dtype=np.float32)

    for i in range(K):
        kpt_x, kpt_y = keypoints_2d[i]
        for y in range(H):
            for x in range(W):
                dx = kpt_x - x
                dy = kpt_y - y
                norm = (dx*dx + dy*dy) ** 0.5 + 1e-8
                gt_vector_field[y, x, i*2]   = dx / norm
                gt_vector_field[y, x, i*2+1] = dy / norm
    return gt_vector_field

def vector_maps_to_patches(vector_maps, patch_count_h, patch_count_w):
    """
    将向量图张量从像素空间转换为Patch空间。

    参数:
    vector_maps (torch.Tensor): 输入的向量图张量，
                                形状为 (B, N_maps, H, W, C)。
    patch_count_h (int): 高度上的Patch数量 (例如 16)。
    patch_count_w (int): 宽度上的Patch数量 (例如 16)。

    返回:
    torch.Tensor: 转换后的Patch特征张量，
                  形状为 (B, N_maps, N_patches, Patch_dim)。
    """
    B, N_maps, H, W, C = vector_maps.shape
    
    # 1. 计算每个Patch的大小
    patch_size_h = H // patch_count_h
    patch_size_w = W // patch_count_w
    
    # print(f"每个Patch的大小为: {patch_size_h}x{patch_size_w} 像素")

    # 2. 关键步骤：通过reshape将H和W维度拆分成 (patch数量, patch大小)
    # 原始: (B, N_maps, H, W, C)
    #      -> (B, N_maps, patch_count_h, patch_size_h, patch_count_w, patch_size_w, C)
    x = vector_maps.view(B, N_maps, patch_count_h, patch_size_h, patch_count_w, patch_size_w, C)

    # 3. 关键步骤：使用permute调整维度顺序，将属于同一个Patch的维度放在一起
    # 目标: 将 patch_size_h, patch_size_w, C 这几个维度相邻，以便后续合并
    # 当前顺序: (B, N_maps, patch_count_h, patch_size_h, patch_count_w, patch_size_w, C)
    # 维度索引:   0, 1,      2,             3,             4,             5,             6
    # 目标顺序: (B, N_maps, patch_count_h, patch_count_w, patch_size_h, patch_size_w, C)
    # 维度索引:   0, 1,      2,             4,             3,             5,             6
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

    # 4. 最后一步：再次reshape，合并Patch数量和Patch内部的维度
    # 当前形状: (B, N_maps, patch_count_h, patch_count_w, patch_size_h, patch_size_w, C)
    #      -> (B, N_maps, patch_count_h * patch_count_w, patch_size_h * patch_size_w * C)
    num_patches = patch_count_h * patch_count_w
    patch_dim = patch_size_h * patch_size_w * C
    
    patches = x.view(B * N_maps, num_patches, patch_dim)
    
    return patches

# @jit(nopyton=True, cache=True) 
def _find_single_keypoint_ransac_refined(k, voter_coords, vector_map_k_vectors, 
                                          num_ransac_iterations=100, distance_threshold=1.5, 
                                          inlier_ratio_threshold=0.40):
    """
    计算单个关键点坐标，并在RANSAC后增加了最小二乘精炼步骤。
    """
    num_voters = len(voter_coords)
    best_hypothesis = np.array([0.0, 0.0], dtype=np.float32)
    max_inliers = -1

    if num_voters < 2:
        return best_hypothesis

    # --- 阶段一: RANSAC 循环 ---
    # (这部分与您之前的代码完全相同)
    for i in range(num_ransac_iterations):
        idx1, idx2 = np.random.choice(num_voters, 2, replace=False)
        p1 = voter_coords[idx1]
        v1 = vector_map_k_vectors[idx1]
        p2 = voter_coords[idx2]
        v2 = vector_map_k_vectors[idx2]

        det = v1[0] * (-v2[1]) - (-v2[0]) * v1[1]
        if abs(det) < 1e-6:
            continue

        b = p2 - p1
        t = (b[0] * (-v2[1]) - b[1] * (-v2[0])) / det
        hypothesis = p1 + t * v1

        vec_p_h = hypothesis - voter_coords
        dot_product = np.sum(vec_p_h * vector_map_k_vectors, axis=1)
        vec_p_h_perp = vec_p_h - dot_product.reshape(-1, 1) * vector_map_k_vectors
        distances = np.sqrt(np.sum(vec_p_h_perp**2, axis=1))
        
        inliers_count = np.sum(distances < distance_threshold)

        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_hypothesis = hypothesis
            
            if max_inliers > (num_voters * inlier_ratio_threshold):
                break
    
    # --- RANSAC循环结束, best_hypothesis 是目前找到的最佳点 ---
    
    
    # --- 阶段二: 结果精炼步骤 (Least-Squares Refinement) ---
    
    # 如果找到的内点太少，直接返回RANSAC的结果，不进行精炼
    if max_inliers < 2:
        return best_hypothesis

    # 1. 根据找到的最佳点，重新确定最终的内点集合
    vec_p_h = best_hypothesis - voter_coords
    dot_product = np.sum(vec_p_h * vector_map_k_vectors, axis=1)
    vec_p_h_perp = vec_p_h - dot_product.reshape(-1, 1) * vector_map_k_vectors
    distances = np.sqrt(np.sum(vec_p_h_perp**2, axis=1))
    inlier_mask = distances < distance_threshold
    
    inlier_coords = voter_coords[inlier_mask]
    inlier_vectors = vector_map_k_vectors[inlier_mask]

    # 至少需要2个内点才能求解
    if len(inlier_coords) < 2:
        return best_hypothesis

    # 2. 构建线性方程组 Ax = b
    #    我们想找一个点 x = [x, y]，它到所有内点射线的距离之和最小
    #    每条射线的方程可以表示为: n_x * x + n_y * y = n·p
    #    其中 n 是射线的法向量, p 是射线上的一点
    
    # a. 计算法向量 A (形状: M, 2)，M是内点数量
    # 法向量 n = [vy, -vx]
    normals = np.zeros_like(inlier_vectors)
    normals[:, 0] = inlier_vectors[:, 1]
    normals[:, 1] = -inlier_vectors[:, 0]
    
    # b. 计算常数项 b (形状: M,)
    # b = n·p
    b_vec = np.sum(normals * inlier_coords, axis=1)

    # 3. 求解正规方程 (A^T * A) * x = (A^T * b)
    try:
        # A^T * A
        ATA = normals.T @ normals
        # A^T * b
        ATb = normals.T @ b_vec
        
        # 使用np.linalg.solve求解，比手动求逆更稳定
        refined_point = np.linalg.solve(ATA, ATb)
        
        return refined_point.astype(np.float32)

    except np.linalg.LinAlgError:
        # 如果矩阵奇异（例如所有内点射线都平行），则无法求解
        # 在这种情况下，我们回退到使用RANSAC的原始结果
        return best_hypothesis
    
import time
# @jit(nopython=True)
def _find_single_keypoint_ransac(k, voter_coords, vector_map_k_vectors, 
                                 num_ransac_iterations=100, distance_threshold=1.5, 
                                 inlier_ratio_threshold=0.40):
    """
    一个经过优化的工作函数，用于计算单个关键点的坐标。
    使用 Numba JIT 编译以获得极致的CPU性能。
    """
    num_voters = len(voter_coords)
    best_hypothesis = np.array([0.0, 0.0], dtype=np.float32)
    max_inliers = -1

    for i in range(num_ransac_iterations):
        # 随机采样2个点
        idx1, idx2 = np.random.choice(num_voters, 2, replace=False)
        p1 = voter_coords[idx1]
        v1 = vector_map_k_vectors[idx1]
        p2 = voter_coords[idx2]
        v2 = vector_map_k_vectors[idx2]

        # 解线性方程组计算交点
        # A = [[v1[0], -v2[0]], [v1[1], -v2[1]]]
        # det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
        det = v1[0] * (-v2[1]) - (-v2[0]) * v1[1]

        if abs(det) < 1e-6: # 线几乎平行
            continue

        # 使用克莱姆法则直接求解，比lstsq更快
        b = p2 - p1
        t = (b[0] * (-v2[1]) - b[1] * (-v2[0])) / det
        hypothesis = p1 + t * v1

        # 计票/评分
        vec_p_h = hypothesis - voter_coords
        dot_product = np.sum(vec_p_h * vector_map_k_vectors, axis=1)
        vec_p_h_perp = vec_p_h - dot_product.reshape(-1, 1) * vector_map_k_vectors
        distances = np.sqrt(np.sum(vec_p_h_perp**2, axis=1))
        
        inliers_count = np.sum(distances < distance_threshold)

        if inliers_count > max_inliers:
            max_inliers = inliers_count
            best_hypothesis = hypothesis
            
            # 提前退出机制
            if max_inliers > (num_voters * inlier_ratio_threshold):
                break
    return best_hypothesis

def find_keypoints_from_vector_map(vector_map, segmentation_mask=None, max_voters=5000, **kwargs):
    """
    优化后的主函数，使用并行处理来加速。
    """
    H, W, C = vector_map.shape
    num_keypoints = C // 2

    if segmentation_mask is None:
        segmentation_mask = np.ones((H, W), dtype=np.uint8)
    
    voter_coords_all = np.argwhere(segmentation_mask > 0)
    voter_coords_all = np.flip(voter_coords_all, axis=1).astype(np.float32) # Numba需要类型明确

    if len(voter_coords_all) < 2:
        return np.zeros((num_keypoints, 2), dtype=np.float32)

    # --- 优化1: 限制投票者最大数量 ---
    if len(voter_coords_all) > max_voters:
        indices = np.random.choice(len(voter_coords_all), max_voters, replace=False)
        voter_coords = voter_coords_all[indices]
    else:
        voter_coords = voter_coords_all

    # 提取所有投票者的所有向量并归一化
    raw_vectors_all = vector_map[voter_coords[:, 1].astype(np.intp), voter_coords[:, 0].astype(np.intp), :]
    norms = np.linalg.norm(raw_vectors_all.reshape(-1, 2), axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit_vectors_all = raw_vectors_all.reshape(-1, 2) / norms
    unit_vectors_all = unit_vectors_all.reshape(len(voter_coords), num_keypoints, 2)

    results = np.zeros((num_keypoints, 2), dtype=np.float32)
    for k in range(num_keypoints):
        # results[k] = _find_single_keypoint_ransac(k, voter_coords, unit_vectors_all[:, k, :])
        results[k] = _find_single_keypoint_ransac_refined(k, voter_coords, unit_vectors_all[:, k, :])
    print(results)
    return np.array(results, dtype=np.float32)

# def find_keypoints_from_vector_map(vector_map, segmentation_mask=None, num_ransac_iterations=100, distance_threshold=1.5):
#     """
#     从单个向量图和掩码中计算所有关键点的坐标。
#     此版本增加了对预测向量的归一化步骤。
#     """
#     H, W, C = vector_map.shape
#     num_keypoints = C // 2
#     keypoints_coords = np.zeros((num_keypoints, 2), dtype=np.float32)

#     if segmentation_mask is None:
#         segmentation_mask = np.ones((H, W), dtype=np.uint8)
#     voter_coords = np.argwhere(segmentation_mask > 0)
#     voter_coords = np.flip(voter_coords, axis=1) # -> (x, y)

#     if len(voter_coords) < 2:
#         return keypoints_coords

#     for k in range(num_keypoints):
#         # 提取当前关键点对应的原始向量
#         raw_vectors = vector_map[voter_coords[:, 1], voter_coords[:, 0], k*2:(k*2)+2]

#         # ==================== 新增的关键步骤：归一化 ====================
#         # 计算每个向量的模长 (L2 Norm)
#         norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
#         # 防止除以零
#         norms[norms == 0] = 1e-8
#         # 得到单位向量
#         unit_vectors = raw_vectors / norms
#         # =============================================================

#         best_hypothesis = None
#         max_inliers = -1

#         for _ in range(num_ransac_iterations):
#             indices = np.random.choice(len(voter_coords), 2, replace=False)
#             # 使用归一化后的单位向量进行计算
#             p1, v1 = voter_coords[indices[0]], unit_vectors[indices[0]]
#             p2, v2 = voter_coords[indices[1]], unit_vectors[indices[1]]
            
#             # ... RANSAC的后续步骤（计算交点、计票）完全不变 ...
#             A = np.array([v1, -v2]).T
#             b = p2 - p1
#             try:
#                 solution = np.linalg.lstsq(A, b, rcond=None)[0]
#                 # solution = line_intersection(p1, v1, p2, v2)
#                 t = solution[0]
#                 hypothesis = p1 + t * v1
#             except np.linalg.LinAlgError:
#                 continue

#             # 使用归一化后的单位向量进行计票
#             vec_p_h = hypothesis - voter_coords
#             vec_v = unit_vectors
            
#             dot_product = np.sum(vec_p_h * vec_v, axis=1)
#             vec_p_h_perp = vec_p_h - dot_product[:, np.newaxis] * vec_v
#             distances = np.linalg.norm(vec_p_h_perp, axis=1)

#             inliers_count = np.sum(distances < distance_threshold)

#             if inliers_count > max_inliers:
#                 max_inliers = inliers_count
#                 best_hypothesis = hypothesis

#             inlier_ratio_threshold = 0.20 # 设定一个80%的内点率阈值
#             if max_inliers > (len(voter_coords) * inlier_ratio_threshold):
#                 # print("提前退出：已找到足够好的解！")
#                 break # 直接跳出 for 循环

#         if best_hypothesis is not None:
#              keypoints_coords[k] = best_hypothesis

#     return keypoints_coords

def line_intersection(p1, v1, p2, v2):
    # 2D 向量叉积
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if abs(cross) < 1e-8:  # 平行
        return None
    t = ((p2 - p1)[0]*v2[1] - (p2 - p1)[1]*v2[0]) / cross
    return p1 + t * v1