import numpy as np
import torch
import trimesh
import math
import contextlib
import io
import os.path as osp
from src.utils.render import nvdiffrast_render, make_mesh_tensors
from PIL import Image

def get_intrinsics(fov_y_deg = 90, width=256, height=256):
    """
    计算相机内参矩阵，给定垂直FOV、图像宽度和高度
    :param fov_y_deg: 垂直FOV（度）
    :param width: 图像的宽度（像素）
    :param height: 图像的高度（像素）
    :return: 内参矩阵 K
    """
    # 将 FOV 转换为弧度
    fov_y_rad = np.deg2rad(fov_y_deg)
    
    # 计算焦距 f_y
    f_y = height / (2 * np.tan(fov_y_rad / 2))
    
    # 计算焦距 f_x，根据宽高比假设焦距相同
    f_x = f_y * (width / height)
    
    # 主点位置 (通常是图像中心)
    c_x = width / 2
    c_y = height / 2
    
    # 相机内参矩阵
    intrinsic_matrix = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])
    
    return intrinsic_matrix

def get_extrinsics(n, radius=10, center=(0, 0, 0)):
    """
    生成n个摄像机的外部参数（变换矩阵）。
    :param n: 渲染图像数量
    :param radius: 摄像机与物体中心的距离（即球面半径）
    :param center: 物体的中心位置
    :return: 外部参数（变换矩阵列表）
    """
    extrinsics = []

    # 将n个摄像机分布在单位球面上
    phi = math.pi * (3.0 - math.sqrt(5.0))  # 黄金角度，用于分布点
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2  # y从1到-1
        radius_at_y = math.sqrt(1 - y * y)  # 计算对应的半径
        x = math.cos(phi * i) * radius_at_y
        z = math.sin(phi * i) * radius_at_y
        
        # 计算摄像机位置
        camera_position = np.array([x * radius, y * radius, z * radius]) + np.array(center)
        
        # 计算旋转矩阵：从相机坐标系到物体坐标系的变换，保证相机朝向物体中心
        forward = -camera_position / np.linalg.norm(camera_position)  # 方向向量（指向物体中心）
        up = np.array([0, 0, -1])  # 世界坐标系中的“上”向量
        right = np.cross(up, forward)  # 右向量
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)  # 重新计算“上”向量，保证正交
        rotation_matrix = np.array([right, up, forward])

        # 变换矩阵（4x4），包含旋转矩阵和平移向量
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = rotation_matrix
        extrinsic_matrix[:3, 3] = -np.dot(rotation_matrix, camera_position)  # 平移向量：物体坐标系到相机坐标系的平移
        
        extrinsics.append(extrinsic_matrix)

    return np.array(extrinsics)

def get_reference_images(model_path, ref_image_size, ref_image_num, radius=2.5, center=(0, 0, 0), fov_y_deg=75, device='cuda', save_path=None):
    # Use the normalized mesh
    if not 'model_n' in model_path:
        model_path = model_path.replace("model/", "model_n/")

    if not model_path.endswith(".obj"):
        model_path = osp.join(model_path, "meshes/model.obj")
    # Load object mesh model
    mesh = trimesh.load(model_path)

    size = mesh.bounding_box.extents
    size = np.repeat(size[np.newaxis, :], ref_image_num, axis=0)

    # Generate n views
    n = ref_image_num
    w, h = ref_image_size[1], ref_image_size[0]
    extrinsics = get_extrinsics(n, radius, center)
    intrinsics = get_intrinsics(fov_y_deg=fov_y_deg, width=w, height=h)

    # Render n views 
    ob_in_cams = torch.as_tensor(extrinsics, device=device, dtype=torch.float)
    # print(ob_in_cams)
    # K = torch.as_tensor(intrinsics, device='cuda', dtype=torch.float)
    mesh_tensors = make_mesh_tensors(mesh, device=device)
    color, depth, normal_map = nvdiffrast_render(K=intrinsics, H=h, W=w, ob_in_cams=ob_in_cams, device=device, context='cuda', mesh_tensors=mesh_tensors, use_light=True)
    if save_path is not None:
        colors = color.data.cpu().numpy()
        for i, color in enumerate(colors):
            image = Image.fromarray((color*255).astype(np.uint8))
            image.save(f'{save_path}/ref{str(i)}.png')
    # K = torch.zeros([4, 4])
    # K[:3, :3] = torch.from_numpy(intrinsics)
    # K[2, 2] = 0
    # K[3, 2] = 1
    # K[2, 3] = 1
    # K = K.unsqueeze(0).repeat(ref_image_num, 1, 1)
    intrinsics = np.repeat(intrinsics[np.newaxis, :, :], ref_image_num, axis=0)
    return color, extrinsics, intrinsics, size

if __name__ == '__main__':
    import os
    model_path = "datasets/gso/model_n/QHPomegranate/meshes/model.obj"
    save_path = "QHPomegranate"
    if not osp.exists(save_path):
        os.makedirs(save_path)
    ref_image_size = [512, 512]
    ref_image_num = 20
    get_reference_images(model_path, ref_image_size, ref_image_num, radius=3, fov_y_deg=45, save_path=save_path)