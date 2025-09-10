import os
import json
import torch
import cv2
import numpy as np
import time

import albumentations as A

from utils.geometry import matrix_to_rot6d_numpy, translation_parameterization, compute_normalization_scale, warp_intrinsics, project_3d_box_to_2d
from .utils import load_and_preprocess_images, load_images, get_transform, images_augment
from src.utils.pixel_voting import generate_pvmap_wo_mask, generate_pvmap_batch, generate_pvmap_batch_numba, generate_all_pvmaps_numba_serial

class Onepose(torch.utils.data.Dataset):
    def __init__(self, config, transform=None):
        super().__init__()
        self.config = config
        self.width, self.height = config.data.image_size
        self.vis_query = False

        self.data_list = self.read_data_list(config)

        self.target_size = self.config.data.target_size

        if self.config.data.augment:
            self.transform = get_transform(config)
        else:
            self.transform = None   

    def read_data_list(self, config):
        data_list = []
        
        if os.path.exists(config.data.data_json):
            with open(config.data.data_json, 'r') as f:
                data_list = json.load(f)
            return data_list
        else:
            data_root = config.data.data_root
            assert os.path.exists(data_root), f"Data root directory {data_root} does not exist."

            # Iterate over object sequences
            object_sequences = os.listdir(data_root)
            for obj_sequence in object_sequences:
                # Normalize the object scale
                bbox_file = os.path.join(data_root, obj_sequence, 'box3d_corners.txt')
                if not os.path.exists(bbox_file):
                    print(f"Object scale file {bbox_file} does not exist.")
                    continue
                else:
                    bbox = np.loadtxt(bbox_file)
                    scale = compute_normalization_scale(bbox)
                obj_sequence_path = os.path.join(data_root, obj_sequence)

                if not os.path.isdir(obj_sequence_path):
                    continue
                sequences = os.listdir(obj_sequence_path)
                sequences = [s for s in sequences if '-' in s]  # Filter sequences with '-'
                 # Use the first sequence as reference
                sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(obj_sequence_path, sequence))]
                sequences.sort(key=lambda x: int(x.split('-')[-1]))
                ref_sequence = sequences[0]
                sequences = sequences[1:]

                ref_sequence_path = os.path.join(obj_sequence_path, ref_sequence)

                for sequence in sequences:
                    sequence_path = os.path.join(obj_sequence_path, sequence)
                    if not os.path.isdir(sequence_path):
                        continue

                    # Read the sequence data
                    sequence_data_list = read_sequence_data(sequence_path, ref_sequence_path, interval=self.config.data.interval, obj_scale=scale)
                    if sequence_data_list is not None:
                        data_list.extend(sequence_data_list)
            
            # Write the data list to a JSON file
            os.makedirs(os.path.dirname(config.data.data_json), exist_ok=True)
            with open(config.data.data_json, 'w') as f:
                json.dump(data_list, f, indent=4)

            return data_list

    def sample_ref_data(self, ref_sequence_path):
        num_ref_images = self.config.data.num_ref_images
        ref_images = []

        ref_color_path = os.path.join(ref_sequence_path, 'color')
        ref_color_files = sorted(os.listdir(ref_color_path), key=lambda x: int(x.split('.')[0]))

        # Sample reference images at regular intervals
        if self.config.data.random_ref:
            ref_data_list = read_reference_data(ref_sequence_path, num_ref_images)
        else:
            interval = max(1, len(ref_color_files) // (num_ref_images - 1))
            ref_data_list = read_sequence_data(ref_sequence_path, interval=interval)

        return ref_data_list
    
    def read_query_image(self, color_path):
        if not os.path.exists(color_path):
            raise FileNotFoundError(f"Color image {color_path} does not exist.")
        
        # Load the image
        if self.config.data.augment:
            images = load_images([color_path])
            return images_augment(images, self.transform, self.config)
        else:
            return load_and_preprocess_images([color_path], self.config.data.target_size)
    
    def read_ref_images(self, color_path_list):
        # Load the image
        if self.config.data.augment:
            # t1 = time.time()
            images = load_images(color_path_list)
            # t2 = time.time()
            # print("Open time: ", t2 - t1)
            return images_augment(images, self.transform, self.config)
        else:
            return load_and_preprocess_images(color_path_list, self.config.data.target_size)

    # def pose_parameterization(self, pose)

    def gen_labels(self, 
                   data, 
                   ref_data_list,
                   query_crop_center,
                   query_crop_shape,
                   query_zoom_ratio,
                   ref_crop_centers,
                   ref_crop_shapes,
                   ref_zoom_ratios,
                   obj_scale=1,
        ):
        query_pose = np.array(np.loadtxt(data['pose']), dtype=np.float32)
        query_intrinsics = np.array(np.loadtxt(data['intrinsics']), dtype=np.float32)
        # Warp intrinsics according to the crop bbox
        query_intrinsics = warp_intrinsics(query_intrinsics[None, ...], query_crop_center, query_crop_shape, query_zoom_ratio)

        ref_poses = []
        ref_intrinsics = []
        for ref_data in ref_data_list:
            ref_poses.append(np.loadtxt(ref_data['pose']))
            ref_intrinsics.append(np.loadtxt(ref_data['intrinsics']))
        ref_poses = np.array(ref_poses, dtype=np.float32)
        ref_intrinsics = np.array(ref_intrinsics, dtype=np.float32)
        # Warp intrinsics according to the crop bbox
        ref_intrinsics = warp_intrinsics(ref_intrinsics, ref_crop_centers, ref_crop_shapes, ref_zoom_ratios)

        # Read 3D bounding box
        bbox_3d_file = os.path.dirname(data['ref_sequence']) + '/box3d_corners.txt'
        bbox_3d = np.loadtxt(bbox_3d_file)*0.8
        # Project 3D bounding box to 2D image
        query_ref_poses = np.concatenate((query_pose[None, ...], ref_poses), axis=0)
        query_ref_intrinsics = np.concatenate((query_intrinsics, ref_intrinsics), axis=0)
        bbox_2d = project_3d_box_to_2d(bbox_3d, query_ref_poses, query_ref_intrinsics)
        
        # Generate GT Point-Voting map
        # pv_maps = generate_pvmap_batch(bbox_2d, [self.target_size, self.target_size])
        # pv_maps = generate_pvmap_batch_numba(bbox_2d, [self.target_size, self.target_size])
        pv_maps = []
        for i in range(query_ref_poses.shape[0]):
            pv_map = generate_pvmap_wo_mask(bbox_2d[i], [self.target_size, self.target_size])
            pv_maps.append(pv_map)
        pv_maps = np.array(pv_maps, dtype=np.float32)

        pv_maps_offset = pv_maps[:1] - pv_maps

        # Normalize 2D bounding box
        bbox_2d = bbox_2d / self.target_size 
        query_info = {
            'query_pose': np.array(query_pose[None, ...], dtype=np.float32), #query_pose[None, ...],
            'query_intrinsics': np.array(query_intrinsics, dtype=np.float32), #query_intrinsic,
            'query_crop_center': np.array(query_crop_center, dtype=np.float32), #query_crop_center,
            'query_crop_shape': np.array(query_crop_shape, dtype=np.float32), #query_crop_shape,
            'query_zoom_ratio': np.array(query_zoom_ratio, dtype=np.float32), #query_zoom_ratio,
            'query_shape': np.array([self.width, self.height], dtype=np.float32),
            'bbox_2d': bbox_2d[:1],
            'obj_scale': np.array([obj_scale, obj_scale, obj_scale], dtype=np.float32),
        }
        ref_info = {
            'ref_poses': np.array(ref_poses, dtype=np.float32), #ref_poses,
            'ref_intrinsics': np.array(ref_intrinsics, dtype=np.float32), #ref_intrinsics,
            'ref_crop_centers': np.array(ref_crop_centers, dtype=np.float32), #ref_crop_centers,
            'ref_crop_shapes': np.array(ref_crop_shapes, dtype=np.float32), #ref_crop_shapes,
            'ref_zoom_ratios': np.array(ref_zoom_ratios, dtype=np.float32), #ref_zoom_ratios,
            'pv_maps': np.array(pv_maps[1:], dtype=np.float32), #pv_maps[1:],
            'bbox_2d': bbox_2d[1:],
            'bbox_3d': bbox_3d,
            'ref_shapes': np.array([self.width, self.height], dtype=np.float32).repeat(len(ref_poses), axis=0),
        }

        pv_maps_input = np.concatenate((np.zeros((1, self.target_size, self.target_size, 16)), pv_maps[1:]), axis=0)

        return pv_maps_offset, pv_maps_input, query_info, ref_info

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        t1 = time.time()
        data = self.data_list[idx]
        
        color = data['color']
        ref_sequence = data['ref_sequence']
        obj_scale = data['obj_scale']
        
        # Sample reference images
        ref_data_list = self.sample_ref_data(ref_sequence)

        # Load the image
        query_image, query_bbox_center, query_bbox_shape, query_zoom_ratio = self.read_query_image(color)
        # ref_images, ref_zoom_centers, ref_zoom_ratios = load_and_preprocess_images(
        #     [ref_data['color'] for ref_data in ref_data_list],
        #     self.config.data.target_size,
        # )
        ref_images, ref_bbox_centers, ref_bbox_shapes, ref_zoom_ratios = self.read_ref_images([ref_data['color'] for ref_data in ref_data_list])
        query_ref_images = np.concatenate((query_image, ref_images), axis=0)
        t2 = time.time()
        # print("Image loading and preprocessing time: {:.4f} s".format(t2 - t1))
        # Generate the ground truth labels
        pv_maps_offset, pv_maps_input, query_info, ref_info = self.gen_labels(data, ref_data_list, 
                                                                    query_bbox_center, query_bbox_shape, query_zoom_ratio, 
                                                                    ref_bbox_centers, ref_bbox_shapes, ref_zoom_ratios, obj_scale)
        
        data_dict = {
            'images': torch.from_numpy(query_ref_images).float(),
            'pv_maps_offset': torch.from_numpy(pv_maps_offset).float(),
            'query_info': query_info,
            'ref_info': ref_info,
            'pv_maps_input': torch.from_numpy(pv_maps_input).float(),
        }

        if self.vis_query:
            import cv2
            query_image_vis = query_image[0]
            query_2d_bbox = query_info['bbox_2d'][0]
            def denormalize_img(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                """
                反归一化图像张量，输入 tensor 为 (3, H, W)
                """
                mean = np.array(mean).reshape(3, 1, 1)
                std = np.array(std).reshape(3, 1, 1)
                return ((img * std + mean)*255).astype(np.uint8)
            denorm_img = denormalize_img(query_image_vis)
            denorm_img = denorm_img.transpose(1, 2, 0)
            for bbox in query_2d_bbox:
                cv2.circle(denorm_img, (int(bbox[0]*self.target_size), int(bbox[1]*self.target_size)), 5, (0, 255, 0), 2)
            # cv2.circle(denorm_img, (100, 100), 10, (0, 0, 255), 2)
            cv2.imwrite('query.png', cv2.cvtColor(denorm_img, cv2.COLOR_RGB2BGR))
        t3 = time.time()
        # print("Data loading and processing time: {:.4f} s".format(t3 - t1))
        return data_dict

import random
def read_reference_data(ref_sequence_path, num_reference_images=20, obj_scale=1):
    ref_data_list = []
    color_path = os.path.join(ref_sequence_path, 'color')
    color_files = sorted(os.listdir(color_path), key=lambda x: int(x.split('.')[0]))
    color_files_sampled = random.sample(color_files, num_reference_images)
    for file in color_files_sampled:
        if file.endswith('.jpg') or file.endswith('.png'):
            color_file = os.path.join(color_path, file)
            intrinsics_file = os.path.join(ref_sequence_path, 'intrin_ba', file.replace('.jpg', '.txt').replace('.png', '.txt'))
            pose_file = os.path.join(ref_sequence_path, 'poses_ba', file.replace('.jpg', '.txt').replace('.png', '.txt'))
            data = {
                'color': color_file,
                'intrinsics': intrinsics_file,
                'pose': pose_file,
                'ref_sequence': ref_sequence_path,
            }
            ref_data_list.append(data)

    return ref_data_list


def read_sequence_data(sequence_path, ref_sequence_path=None, interval=1, obj_scale=1):
    sequence_data_list = []
    color_path = os.path.join(sequence_path, 'color')
    color_files = sorted(os.listdir(color_path), key=lambda x: int(x.split('.')[0]))
    for file in color_files[::interval]:
        if file.endswith('.jpg') or file.endswith('.png'):
            color_file = os.path.join(color_path, file)
            intrinsics_file = os.path.join(sequence_path, 'intrin_ba', file.replace('.jpg', '.txt').replace('.png', '.txt'))
            pose_file = os.path.join(sequence_path, 'poses_ba', file.replace('.jpg', '.txt').replace('.png', '.txt'))
            data = {
                'color': color_file,
                'intrinsics': intrinsics_file,
                'pose': pose_file,
                'ref_sequence': ref_sequence_path,
                'obj_scale': obj_scale
            }
            sequence_data_list.append(data)

    return sequence_data_list


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config_file = "configs/train_demo_debug.yaml"
    config = OmegaConf.load(config_file)
    dataset = Onepose(config.val)

    for i in range(len(dataset)):
        data = dataset[i]
        print(data)