import numpy as np
import multiprocessing
from functools import partial
import math
import time
# 推荐安装numba库: pip install numba
from numba import jit, njit, prange

def generate_pvmap_optimized(keypoints_2d, output_shape):
    """
    高效地、无掩码地批量生成PVNet向量场真值。
    此函数完全向量化，以替代多层嵌套的Python for循环。

    参数:
    keypoints_2d (np.array): 关键点坐标，形状为 (K, 2)，K是关键点数量。
    output_shape (tuple): (H, W)，输出向量图的尺寸。

    返回:
    np.array: 形状为 (H, W, K*2) 的密集向量图真值数组。
    """
    H, W = output_shape
    K = keypoints_2d.shape[0]

    # 1. 创建坐标网格（只需一次）
    # y_coords, x_coords 形状均为 (H, W)
    y_coords, x_coords = np.indices((H, W), dtype=np.float32)

    # 2. 准备广播 (核心技巧)
    # 将关键点坐标的维度进行扩展，以便它们可以与整个像素坐标网格进行运算。
    # kpts_x 的形状: (K,) -> (K, 1, 1)
    kpts_x = keypoints_2d[:, 0][:, np.newaxis, np.newaxis]
    # kpts_y 的形状: (K,) -> (K, 1, 1)
    kpts_y = keypoints_2d[:, 1][:, np.newaxis, np.newaxis]
    
    # 3. 一次性计算所有原始向量
    # (K, 1, 1) - (H, W) -> 广播成 (K, H, W)
    raw_vx = kpts_x - x_coords
    raw_vy = kpts_y - y_coords

    # 4. 一次性计算所有模长并进行归一化
    # 所有操作都是在 (K, H, W) 的张量上进行的，非常高效
    norm = np.sqrt(raw_vx**2 + raw_vy**2) + 1e-8
    unit_vx = raw_vx / norm
    unit_vy = raw_vy / norm

    # 5. 将 vx 和 vy 堆叠并重塑为最终的 (H, W, K*2) 格式
    # a. 堆叠成 (K, H, W, 2)，最后一个维度代表 (vx, vy)
    stacked_vectors = np.stack([unit_vx, unit_vy], axis=-1)
    
    # b. 调整维度顺序为 (H, W, K, 2)
    transposed_vectors = stacked_vectors.transpose(1, 2, 0, 3)
    
    # c. 最后，将最后两个维度 (K, 2) 合并为 K*2
    gt_vector_field = transposed_vectors.reshape(H, W, K * 2)
    
    return gt_vector_field

@njit(parallel=True)
def generate_pvmap_wo_mask_slow(keypoints_2d, output_shape):
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

if __name__ == '__main__':
    while True:
        # --- 准备测试数据 ---
        H, W = 224, 224
        K = 8
        keypoints = np.random.rand(K, 2) * W

        # --- 测试优化后的快速函数 ---
        print("\n正在运行优化后的函数...")
        start_time = time.time()
        result_fast = generate_pvmap_optimized(keypoints, (H, W))
        end_time = time.time()
        print(f"优化函数耗时: {end_time - start_time:.4f} 秒")

        # --- 测试您的原始慢速函数 ---
        print("正在运行您的原始函数 (可能需要一些时间)...")
        start_time = time.time()
        result_slow = generate_pvmap_wo_mask_slow(keypoints, (H, W))
        end_time = time.time()
        print(f"原始函数耗时: {end_time - start_time:.4f} 秒")

        # print(result_slow)
        # print(result_fast)

        # --- 验证结果是否一致 ---
        # np.allclose 用于比较两个浮点数数组是否在容差范围内近似相等
        are_results_same = np.allclose(result_slow, result_fast)
        print(f"\n两个函数的结果是否一致: {are_results_same}")