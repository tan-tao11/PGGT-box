import trimesh
import argparse
import os
import os.path as osp
import shutil
import numpy as np
from trimesh.transformations import translation_matrix, scale_matrix
from tqdm import tqdm

parser = argparse.ArgumentParser(
        description='Normalize all meshes in a directory')
parser.add_argument('--input_dir', '-i', type=str, default='/data/tantao/my_methods/diffusion_pose/datasets/gso/model')
parser.add_argument('--output_dir', '-o', type=str, default='/data/tantao/my_methods/diffusion_pose/datasets/gso/model_n')

args = parser.parse_args()

input_dir = args.input_dir
obj_folders = os.listdir(input_dir)

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)

for obj_folder in tqdm(obj_folders):
    try:
        obj_path = osp.join(input_dir, obj_folder, 'meshes/model.obj')

        # Load object mesh model
        mesh = trimesh.load(obj_path)

        # Get the bounding box of the mesh
        mesh_bb = mesh.bounding_box
        
        # Transform the coordinates origin to the object center
        center = mesh_bb.centroid   
        translation = translation_matrix(-center)  # Create translation matrix
        mesh.apply_transform(translation)  # Apply translation to center the mesh

        # Calculate the scaling factor
        extents = mesh_bb.extents
        longest_side = max(extents)
        scale_factor = 2 / longest_side

        # Apply the scaling factor to normalize the mesh
        scale = scale_matrix(scale_factor)  # Create scale matrix
        mesh.apply_transform(scale)  # Apply scaling to normalize the mesh

        # Get the bounding box of the normalized mesh
        mesh_bb = mesh.bounding_box
        vertices = mesh_bb.vertices
        # 计算中心
        center = vertices.mean(axis=0)

        # 偏移
        offsets = vertices - center

        # 用符号编码 (x>0,y>0,z>0) 来分组
        codes = np.array([(ox > 0, oy > 0, oz > 0) for ox, oy, oz in offsets], dtype=int)

        # 建立一个字典，key=符号组合，value=顶点
        corner_dict = {tuple(code): v for code, v in zip(codes, vertices)}

        # 按照你给的顺序提取
        ordered_vertices = np.array([
            corner_dict[(0,0,0)],  # (-,-,-)
            corner_dict[(0,0,1)],  # (-,-,+)
            corner_dict[(0,1,1)],  # (-,+,+)
            corner_dict[(0,1,0)],  # (-,+,-)
            corner_dict[(1,0,0)],  # (+,-,-)
            corner_dict[(1,0,1)],  # (+,-,+)
            corner_dict[(1,1,1)],  # (+,+,+)
            corner_dict[(1,1,0)]   # (+,+,-)
        ])

        # Save the normalized mesh
        normalized_mesh_path = osp.join(args.output_dir, obj_folder, 'meshes/model.obj')
        os.makedirs(osp.dirname(normalized_mesh_path), exist_ok=True)
        mesh.export(normalized_mesh_path)
        np.savetxt(osp.join(args.output_dir, obj_folder, 'meshes/box3d_corners.txt'), ordered_vertices, fmt='%f')
        shutil.copy(osp.join(input_dir, obj_folder, 'materials/textures/texture.png'), osp.join(args.output_dir, obj_folder, 'meshes/texture.png'))
        shutil.copy(osp.join(input_dir, obj_folder, 'meshes/model.mtl'), osp.join(args.output_dir, obj_folder, 'meshes/material.mtl'))
    except:
        continue
