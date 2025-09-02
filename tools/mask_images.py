import cv2
import os
import numpy as np
from tqdm import tqdm

# 文件夹路径
image_dir = '/data3/tantao/dataspace/LineMOD/lm/test/000001/rgb'
mask_dir = '/data3/tantao/dataspace/LineMOD/lm/test/000001/mask'
output_dir = '/data3/tantao/my_methods/6D_vggt/data/lm/test/000001/mask_images'

os.makedirs(output_dir, exist_ok=True)

# 获取图像文件列表
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

for fname in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname.split('.')[0] + '_000000.png')  
    output_path = os.path.join(output_dir, fname)

    # 确保掩码存在
    if not os.path.exists(mask_path):
        print(f"Warning: No mask found for {fname}, skipping.")
        continue

    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error reading {fname}, skipping.")
        continue

    # 生成前景掩码
    foreground_mask = mask > 0
    foreground_mask_3c = np.repeat(foreground_mask[:, :, np.newaxis], 3, axis=2)

    # 去除背景
    output_image = np.zeros_like(image)
    output_image[foreground_mask_3c] = image[foreground_mask_3c]

    # 保存结果
    cv2.imwrite(output_path, output_image)