import torch
import os, json, re
import imageio
import pickle
import torch.distributed as dist
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf
from PIL import Image
from .utils import to_homogeneous
from .reference_images_process_gso import get_reference_images
from .utils import load_and_preprocess_images, load_images, get_transform, images_augment
from utils.geometry import matrix_to_rot6d_numpy, translation_parameterization, compute_normalization_scale, warp_intrinsics, project_3d_box_to_2d
import torch.distributed as dist


def obj_name_match(obj_name, obj_list):
    for obj in obj_list:
        if re.sub(r'\d+', '', obj).lower() in obj_name.lower():
            return obj

def enlarge_bbox(bbox, k):
    x1, y1, x2, y2 = bbox
    # 计算边界框的中心
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 计算新的宽度和高度
    new_width = (x2 - x1) * k
    new_height = (y2 - y1) * k

    # 计算新的边界框坐标
    new_x1 = cx - new_width / 2
    new_y1 = cy - new_height / 2
    new_x2 = cx + new_width / 2
    new_y2 = cy + new_height / 2
    new_bbox = [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]
    return new_bbox


class Gso(Dataset):
    def __init__(self, config, device):
        self.config = config
        self.width, self.height = config.data.image_size
        self.vis_query = False
        self.data_root = config.data1.data_root
        self.device = device
        self.target_size = self.config.data.target_size
        self.num_sequences = self.config.data.num_ref_images

        if self.config.data1.augment:
            self.transform = get_transform(config)

        self.obj_list = self.get_obj_list()
        self.frame_list = self.get_frame_list()
    
    def get_obj_list(self):
        obj_list = []
        data_names = self.config.data1.data_name
        try:
            for data_name in data_names:
                obj_model_root = osp.join(self.data_root, data_name, 'model_n')
                obj_list.extend(os.listdir(obj_model_root))
        except:
            obj_model_root = osp.join(self.data_root, 'model_n')
            obj_list.extend(os.listdir(obj_model_root))
        return obj_list

    def get_intrinsic(self, path):
        with open(path,'r') as ff:
            camera_params = json.load(ff)
        world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
        W, H = camera_params["renderProductResolution"]
        glcam_in_cvcam = np.array([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]]).astype(float)
        cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
        world_in_cam = np.linalg.inv(cam_in_world)
        focal_length = camera_params["cameraFocalLength"]
        horiz_aperture = camera_params["cameraAperture"][0]
        vert_aperture = H / W * horiz_aperture
        focal_y = H * focal_length / vert_aperture
        focal_x = W * focal_length / horiz_aperture
        center_y = H * 0.5
        center_x = W * 0.5

        fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
        K = np.eye(3)
        K[0,0] = fx
        K[1,1] = fy
        K[0,2] = cx
        K[1,2] = cy

        return K, world_in_cam    
    
    def get_obj_pose(self, obj_in_world, world_in_cam):
        def normalizeRotation(pose):
            new_pose = pose.copy()
            scales = np.linalg.norm(pose[:3,:3],axis=0)
            new_pose[:3,:3] /= scales.reshape(1,3)
            return new_pose

        obj_in_world = normalizeRotation(obj_in_world)

        obj_in_world = world_in_cam@obj_in_world
        return obj_in_world

    def scene_preprocess(self):
        """
        Fetch all scenes in the dataset and save them into a dictionary, 
        where the key is the scene id and the value is a dictionary containing 
        object name, pose, scale, bounding box, camera intrinsic matrix, and 
        occlusion rate.
        """
        print("Fetching all scenes...")
        data_names = self.config.data1.data_name
        scene_dict = {}
        scene_id = 0

        scenes_path = osp.join(self.data_root, 'scenes')
        sequences = os.listdir(scenes_path)
        for sequence in tqdm(sequences[:self.num_sequences]):
            print(sequence)
            sequence_path = osp.join(scenes_path, sequence, sequence)
            if osp.isdir(sequence_path):
                scenes = os.listdir(sequence_path)
                for scene in tqdm(scenes):
                    # Load states.json
                    with open(osp.join(sequence_path, scene, 'states.json'), 'r') as f:
                        state_data = json.load(f)
                    objects = state_data.get('objects', {})
                    
                    # Extract scene folder
                    scene_folders = [item for item in os.listdir(osp.join(sequence_path, scene)) if item.startswith('scene-')]
                    if not scene_folders:
                        continue
                    scene_folder = scene_folders[0]
                    scene_path = osp.join(sequence_path, scene, scene_folder)
                    
                    # Process scene frames
                    scene_frames = [item for item in os.listdir(scene_path) if item.startswith("Render")]
                    for frame in scene_frames:
                        try:
                            # Find bounding box and camera information
                            bbox_npy_path = osp.join(scene_path, frame, "bounding_box_2d_loose/bounding_box_2d_loose_000000.npy")
                            bbox_json_path = osp.join(scene_path, frame, "bounding_box_2d_loose/bounding_box_2d_loose_labels_000000.json")
                            cam_info_json = osp.join(scene_path, frame, "camera_params/camera_params_000000.json")
                            
                            # load camera
                            K, world_in_cam = self.get_intrinsic(cam_info_json)

                            # load bounding box
                            bounding_boxes = np.load(bbox_npy_path)
                            with open(bbox_json_path, 'r') as f:
                                bbox_labels = json.load(f)
                            
                            for index, bbox in enumerate(bounding_boxes):
                                # Find the object name that matches the bounding box label
                                bbox_class = bbox_labels.get(str(index), {}).get('class', '')
                                matched_obj_name = obj_name_match(bbox_class, self.obj_list)
                                if matched_obj_name is None:
                                    continue
                                x1, y1, x2, y2, occlusion = bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
                                bbox = np.array([x1, y1, x2, y2])
                                # Bbox's width and height must > 0
                                if x2 - x1 <= 0 or y2 - y1 <= 0:
                                    continue

                                # Get object info
                                object_info = objects.get(matched_obj_name, {})
                                if len(object_info) < 1:
                                    continue
                                obj_in_world = np.array(object_info.get("transform_matrix_world", {})).reshape(4,4).T
                                # Get object pose
                                obj_pose = self.get_obj_pose(obj_in_world, world_in_cam)
                                # Get scale
                                pose_scale = np.array(object_info.get("scale", {}))
                                # Create a dictionary for the object in the scene
                                object_in_scene = {
                                    "obj_name": matched_obj_name,
                                    "rgb": osp.join(scene_path, frame, "rgb/rgb_000000.png"),
                                    "obj_pose": obj_pose.tolist(),
                                    "obj_scale": pose_scale.tolist(),
                                    "obj_bbox": bbox.tolist(),
                                    "cam_K": K.tolist(),
                                    "occlusion_rate": float(occlusion),
                                    "obj_model": osp.join(self.data_root, "model_n", matched_obj_name),
                                    }
                                # Add the object to the scene dictionary
                                scene_dict[str(scene_id)] = object_in_scene
                                scene_id += 1
                        except:
                                pass    

        # Save scene dict
        with open(self.config.data1.data_dict, 'w') as f:
            json.dump(scene_dict, f, indent=4)
        print("Fetched all scene!")
        print(f"Total scene: {len(scene_dict)}.")

    def get_frame_list(self):
        scene_json = self.config.data1.data_dict
        while(not osp.exists(scene_json)):
            if self.device == torch.device("cuda:0"):
                self.scene_preprocess()
            
        # dist.barrier()

        with open(scene_json, "r") as f:
            scenes = json.load(f)
        frame_list = list(scenes.values())
        if self.config.train.occlusion_threshold < 1.0:
            frame_list = [frame for frame in frame_list if frame["occlusion_rate"] < self.config.train.occlusion_threshold]
        if self.config.data1.get('obj_list') is not None:
            frame_list = [frame for frame in frame_list if frame["obj_name"] in self.config.data1.obj_list]
        print(f"Totle data number: {len(frame_list)}")
        return frame_list

    def augment_image(self, rgb):
        # Load the image
        return images_augment(rgb, self.transform, self.config)
        # else:
        #     return load_and_preprocess_images([color_path], self.config.data.target_size)
    
    def crop_and_resize_rgb(self, rgb_file, bbox, K=None):
        """
        Crop rgb image based on the given bounding box.

        Args:
            rgb_file (str): Path to rgb image.
            bbox (list): Bounding box coordinates in the format of [x1, y1, x2, y2].

        Returns:
            Image: Cropped rgb image.
        """
        img = load_images([rgb_file])[0]
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img.shape[2], x_max)
        y_max = min(img.shape[1], y_max)

        if K is not None:
            crop_centers = np.array([[(x_min + x_max) / 2.0, (y_min + y_max) / 2.0]])
            crop_shapes = np.array([[x_max - x_min, y_max - y_min]])
            zoom_ratios = np.array([[self.width / (x_max - x_min), self.height / (y_max - y_min)]])
            crop_K = warp_intrinsics(K[None, ...], crop_centers, crop_shapes, zoom_ratios)
            
        # crop_img = img.crop((x_min, y_min, x_max, y_max))
        crop_img = img[:, y_min:y_max, x_min:x_max].transpose(1, 2, 0)

        # resized_img = crop_img.resize((self.width, self.height))
        resized_img = cv2.resize(crop_img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # normalized_img = np.array(resized_img)
        return resized_img[:, :, :3], crop_K
    def gen_labels(self, 
                   data, 
                   query_pose,
                   query_intrinsic,
                   query_crop_center,
                   query_crop_shape,
                   query_zoom_ratio,
                   ref_poses,
                   ref_intrinsics,
                   ref_crop_centers,
                   ref_crop_shapes,
                   ref_zoom_ratios,
                   obj_scale=None,
        ):
        # Apply scale to translation
        if obj_scale is not None:
            query_pose[:3, 3] = query_pose[:3, 3] /  obj_scale

        # Warp intrinsics according to the crop bbox
        query_intrinsic = warp_intrinsics(query_intrinsic, query_crop_center, query_crop_shape, query_zoom_ratio)

        ref_poses = np.array(ref_poses, dtype=np.float32)
        ref_intrinsics = np.array(ref_intrinsics, dtype=np.float32)
        # Warp intrinsics according to the crop bbox
        ref_intrinsics = warp_intrinsics(ref_intrinsics, ref_crop_centers, ref_crop_shapes, ref_zoom_ratios)

        # Read 3D bounding box
        bbox_3d_file = os.path.join(data['obj_model'], 'meshes', 'box3d_corners.txt')
        bbox_3d = np.loadtxt(bbox_3d_file)
        # Project 3D bounding box to 2D image
        query_ref_poses = np.concatenate((query_pose[None, ...], ref_poses), axis=0)
        query_ref_intrinsics = np.concatenate((query_intrinsic, ref_intrinsics), axis=0)
        bbox_2d = project_3d_box_to_2d(bbox_3d, query_ref_poses, query_ref_intrinsics)
        # Normalize 2D bounding box
        bbox_2d = bbox_2d / self.target_size 

        # Generate 2D bbox offset labels
        bbox_2d_offset = (bbox_2d - bbox_2d[0]).reshape(-1, 16)

        query_info = {
            'query_pose': np.array(query_pose[None, ...], dtype=np.float32), #query_pose[None, ...],
            'query_intrinsics': np.array(query_intrinsic, dtype=np.float32), #query_intrinsic,
            'query_crop_center': np.array(query_crop_center, dtype=np.float32), #query_crop_center,
            'query_crop_shape': np.array(query_crop_shape, dtype=np.float32), #query_crop_shape,
            'query_zoom_ratio': np.array(query_zoom_ratio, dtype=np.float32), #query_zoom_ratio,
            'query_shape': np.array([self.width, self.height], dtype=np.float32),
            'bbox_2d': bbox_2d[:1],
            'obj_scale': np.array(obj_scale, dtype=np.float32),
        }
        ref_info = {
            'ref_poses': np.array(ref_poses, dtype=np.float32), #ref_poses,
            'ref_intrinsics': np.array(ref_intrinsics, dtype=np.float32), #ref_intrinsics,
            'ref_crop_centers': np.array(ref_crop_centers, dtype=np.float32), #ref_crop_centers,
            'ref_crop_shapes': np.array(ref_crop_shapes, dtype=np.float32), #ref_crop_shapes,
            'ref_zoom_ratios': np.array(ref_zoom_ratios, dtype=np.float32), #ref_zoom_ratios,
            'bbox_2d': bbox_2d[1:],
            'bbox_3d': bbox_3d,
            'ref_shapes': np.array([self.width, self.height], dtype=np.float32).repeat(len(ref_poses), axis=0),
        }

        # Construct pose and intrinsics input
        # intrinsics = np.concatenate((query_intrinsics, ref_intrinsics), axis=0)
        # intrinsics = np.array([[intrinsic[0, 0]/self.target_size, intrinsic[1, 1]/self.target_size, intrinsic[0, 2]/self.target_size, intrinsic[1, 2]/self.target_size] for intrinsic in intrinsics])
        # query_ref_rot_mats = np.concatenate((np.eye(3)[None, ...], ref_rot_mats), axis=0)
        # query_ref_rot_6d = matrix_to_rot6d_numpy(query_ref_rot_mats)
        # query_ref_trans_params = np.concatenate((np.array([[0, 0, 0]]), ref_trans_params), axis=0)
        # query_ref_pose_params = np.concatenate((query_ref_trans_params, query_ref_rot_6d), axis=1)
        # pose_intr_input = np.concatenate((intrinsics, query_ref_pose_params), axis=1)

        # Construct 2d bbox input
        bbox_2d_input = np.concatenate((np.zeros((1, 8, 2)), bbox_2d[1:]), axis=0).reshape(-1, 16)
        

        return bbox_2d_offset, bbox_2d_input, query_info, ref_info
    
    def __getitem__(self, index):
        data = self.frame_list[index]
        # Load rgb and crop it
        rgb_file = data['rgb']
        bbox = data["obj_bbox"]
        if self.config.data1.get('crop_ratio') is not None:
            # Enlarge the bounding box by a factor of crop_ratio
            bbox = enlarge_bbox(bbox, self.config.data1.get('crop_ratio'))

        # Load the query image
        query_intrinsic = np.array(data["cam_K"])
        query_pose = np.array(data["obj_pose"])
        crop_rgb, query_intrinsic = self.crop_and_resize_rgb(rgb_file, bbox, query_intrinsic)  # Crop the rgb image based on the bounding box
        query_image, query_bbox_center, query_bbox_shape, query_zoom_ratio = self.augment_image(crop_rgb.transpose(2, 0, 1)[None, ...])

        # Get references
        local_ref = osp.join(data["obj_model"].replace('model_n', 'ref_rendered'))
        if osp.exists(local_ref):
            with open(osp.join(local_ref, 'data.pkl'), "rb") as f:
                local_ref_data = pickle.load(f)
                ref_images = local_ref_data['ref_images']
                ref_poses = local_ref_data["ref_extrinsics"]
                ref_intrinsics = local_ref_data['ref_intrinsics']
                obj_size = local_ref_data['obj_size']
        else:
            with torch.no_grad():
                    ref_images, ref_poses, ref_intrinsics, obj_size = get_reference_images(
                        data["obj_model"], (self.height, self.width), self.config.data.num_ref_images, device=self.device
                    )
            ref_images = (ref_images.cpu().numpy()*255).astype(np.uint8)
            local_ref_data = {
                "ref_images": ref_images,
                "ref_extrinsics": ref_poses,
                "ref_intrinsics": ref_intrinsics,
                "obj_size": obj_size,
            }
            # Save data
            # Several data workers may process the same object
            try:
                os.makedirs(local_ref)
                with open(osp.join(local_ref, 'data.pkl'), "wb") as f:
                    pickle.dump(local_ref_data, f)
            except:
                pass
        
        ref_images, ref_bbox_centers, ref_bbox_shapes, ref_zoom_ratios = self.augment_image(ref_images.transpose(0, 3, 1, 2))

        # Concat the query and reference images
        query_ref_images = np.concatenate((query_image, ref_images), axis=0)
        # Generate the ground truth labels
        bbox_2d_offset_label, bbox_2d_input, query_info, ref_info = self.gen_labels(data, query_pose, query_intrinsic,
                                                                    query_bbox_center, query_bbox_shape, query_zoom_ratio, 
                                                                    ref_poses, ref_intrinsics,
                                                                    ref_bbox_centers, ref_bbox_shapes, ref_zoom_ratios, data["obj_scale"])

        data_dict = {
            'images': torch.from_numpy(query_ref_images).float(),
            'bbox_2d_offset_label': torch.from_numpy(bbox_2d_offset_label).float(),
            'query_info': query_info,
            'ref_info': ref_info,
            'bbox_2d_input': torch.from_numpy(bbox_2d_input).float(),
        }

        # Visualize
        if self.vis_query:
            import cv2
            query_image_vis = query_image[0]
            # ref_images_vis = ref_images[0]
            # query_image_vis = query_image.transpose(2, 0, 1)
            query_2d_bbox = query_info['bbox_2d'][0]
            # ref_2d_bbox = bbox_2d_input[1].reshape(-1, 2)
            def denormalize_img(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
                """
                反归一化图像张量，输入 tensor 为 (3, H, W)
                """
                mean = np.array(mean).reshape(3, 1, 1)
                std = np.array(std).reshape(3, 1, 1)
                return ((img * std + mean)*255).astype(np.uint8)
            denorm_img = denormalize_img(query_image_vis)
            # denorm_img = denormalize_img(ref_images_vis)
            denorm_img = denorm_img.transpose(1, 2, 0)
            for bbox in query_2d_bbox:
                cv2.circle(denorm_img, (int(bbox[0]*self.target_size), int(bbox[1]*self.target_size)), 5, (0, 255, 0), 2)
            # cv2.circle(denorm_img, (100, 100), 10, (0, 0, 255), 2)
            cv2.imwrite('query.png', cv2.cvtColor(denorm_img, cv2.COLOR_RGB2BGR))

        return data_dict

    def __len__(self):
        return len(self.frame_list)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config_file = "configs/train.yaml"
    config = OmegaConf.load(config_file)
    dataset = Gso(config, 'cuda:0')

    for i in range(len(dataset)):
        data = dataset[i]
        print(data)