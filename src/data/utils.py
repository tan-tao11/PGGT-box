import torch
import numpy as np
from PIL import Image
from torchvision import transforms as TF
import albumentations as A
import albumentations.augmentations.crops as F
import cv2

def to_array(np_img):
    """
    Convert an image array (H x W x C, uint8) to a float32 array (C x H x W),
    with pixel values normalized to [0.0, 1.0].

    Args:
        np_img (np.ndarray): Input image array, shape (H, W, C), dtype=uint8.

    Returns:
        np.ndarray: Normalized image, shape (C, H, W), dtype=float32.
    """
    assert np_img.ndim == 3 and np_img.dtype == np.uint8, "Expected (H, W, C), uint8 image"
    np_img = np_img.astype(np.float32) / 255.0           # Normalize to [0, 1]
    np_img = np.transpose(np_img, (2, 0, 1))              # Convert to (C, H, W)
    return np_img

def to_array_wo_normalize(np_img):
    """
    Convert an image array (H x W x C, uint8) to a float32 array (C x H x W),
    with pixel values normalized to [0.0, 1.0].

    Args:
        np_img (np.ndarray): Input image array, shape (H, W, C), dtype=uint8.

    Returns:
        np.ndarray: Normalized image, shape (C, H, W), dtype=float32.
    """
    assert np_img.ndim == 3 and np_img.dtype == np.uint8, "Expected (H, W, C), uint8 image"
    np_img = np.transpose(np_img, (2, 0, 1))              # Convert to (C, H, W)
    return np_img

def load_images(image_path_list):
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    images = []

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")
        img = to_array_wo_normalize(np.array(img))
        images.append(img)
    
    images = np.stack(images)

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if len(images.shape) == 3:
            images = images[None, ...]

    return images

def load_and_preprocess_images(image_path_list, target_size=518, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        np.array: Batched array of preprocessed images with shape (N, 3, H, W)
        np.array: Batched array of centers of the bounding boxes for each image with shape (N, 2)
        np.array: Batched array of shapes of the bounding boxes for each image with shape (N, 2)
        np.array: Batched array of ratios of the bounding boxes w.r.t. zoom-in size with shape (N, 1)

    Raises:
        ValueError: If the input list is empty or if mode is invalid
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    
    images = []
    ratios = []
    bb_centers = []
    shapes = []
    # target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:

        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14
            ratio = np.array([new_width / width, new_height / height])
            c_xy = [width / 2, height / 2]

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_array(np.array(img))

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        if mode == "pad":
            raise NotImplementedError("Padding mode is not implemented in this function.")
    
        shapes.append(np.array((width, height)))
        images.append(img)
        ratios.append(ratio)
        bb_centers.append(c_xy)
    
    images = np.stack(images)
    ratios = np.stack(ratios) 
    bb_centers = np.stack(bb_centers)
    shapes = np.stack(shapes)

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if len(images.shape) == 3:
            images = images[None, ...]

    # Normalize images and reshape for patch embed
    _resnet_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    _resnet_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = (images - _resnet_mean) / _resnet_std

    return images, bb_centers, shapes, ratios

def crop_and_resize(img, x, y, crop_h, crop_w, resize_h, resize_w, interpolation=cv2.INTER_LINEAR):
    cropped = F.crop(img, x, y, crop_h, crop_w)
    resized = cv2.resize(cropped, (resize_w, resize_h), interpolation=interpolation)
    return resized

class RandomResizedCropWithBox(A.RandomResizedCrop):
    # def __init__(self, height, width, scale=(0.5, 1.0), ratio=(0.75, 1.33), interpolation=cv2.INTER_LINEAR, p=1.0):
    #     super().__init__(height=height, width=width, scale=scale, ratio=ratio, interpolation=interpolation, p=p)
    #     self.crop_x = self.crop_y = self.crop_w = self.crop_h = None
        
    def apply(self, img, x=0, y=0, height=0, width=0, **params):
        # 储存裁剪区域坐标
        self.crop_x = x
        self.crop_y = y
        self.crop_w = width
        self.crop_h = height
        return crop_and_resize(img, x, y, height, width, self.height, self.width, self.interpolation)

    def get_transform_init_args_names(self):
        return super().get_transform_init_args_names() + ("crop_x", "crop_y", "crop_w", "crop_h")

    def get_crop_box(self):
        # 返回裁剪框中心坐标 (cx, cy) 和尺寸 (w, h)
        cx = self.crop_x + self.crop_w / 2
        cy = self.crop_y + self.crop_h / 2
        return cx, cy, self.crop_w, self.crop_h
    
def get_transform(config):
    target_size = config.data.target_size

    transform = A.ReplayCompose([
        # ✅ 随机裁剪+缩放
        A.RandomResizedCrop(size=(target_size, target_size), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
        # ✅ 颜色增强
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        # ✅ 模糊 or 噪声
        A.OneOf([
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
            A.GaussNoise(std_range=(0.1, 0.3), p=0.5),
        ], p=0.3),
        # ✅ 遮挡（Cutout）
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(target_size // 10, target_size // 5),
            hole_width_range=(target_size // 10, target_size // 5),
            fill="random_uniform",
            p=0.2
        ),
        # ✅ 标准化 & 转 Tensor
        A.Normalize(mean=(0.485, 0.456, 0.406),  # DINOv2 官方 mean/std
                std=(0.229, 0.224, 0.225)),
        ])

    return transform

def images_augment(images, transform, config):
    """
    Args: 
        images: (N, C, H, W)
        transform: A.Compose
    """
    target_size = config.data.target_size
    num_frames, w, h = images.shape[0], images.shape[3], images.shape[2]

    if images.shape[1] == 1 or images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
        
    # Apply transform to each frame
    ratios = []
    bb_centers = []
    bb_shapes = []

    transformed_frames = []
    for f in range(num_frames):
        transformed_frame = transform(image=images[f])
        transformed_frames.append(transformed_frame["image"].transpose(2, 0, 1))

        # 获取裁剪框
        bbox = transformed_frame['replay']['transforms'][0]['params']['crop_coords']
        cx, cy, w, h = voc_to_cxcywh(bbox)
        bb_centers.append(np.array([cx, cy]))
        bb_shapes.append(np.array([w, h]))
        ratios.append(np.array([target_size/w, target_size/h]))

    transformed_frames = np.stack(transformed_frames)
    bb_centers = np.stack(bb_centers)
    bb_shapes = np.stack(bb_shapes)
    ratios = np.stack(ratios)

    return transformed_frames, bb_centers, bb_shapes, ratios
    
def voc_to_cxcywh(bbox):
    """
    将 Pascal VOC 格式的 bbox [x_min, y_min, x_max, y_max]
    转换为中心点加尺寸格式 [cx, cy, w, h]
    """
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    return [cx, cy, w, h]

def to_homogeneous(points):
    # points 是形状为 (n, 3) 的点云坐标
    n = points.shape[0]
    
    # 创建一个形状为 (n, 1) 的列向量，所有值为 1
    ones = np.ones((n, 1))
    
    # 在点云坐标后添加一个列，值为 1
    homogeneous_points = np.hstack((points, ones))
    
    return homogeneous_points
    