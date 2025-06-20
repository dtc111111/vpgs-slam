import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio
import trimesh
from tqdm import tqdm
from PIL import Image

# from src.DepthLab.infer import input_image, depth_numpy
from src.utils.eval_utils import mkdir_p
import pykitti
from scipy.interpolate import griddata as interp_grid

OPENCV2DATASET = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
)

import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
import cv2
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from src.DepthLab.src.models.unet_2d_condition import UNet2DConditionModel
from src.DepthLab.src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.DepthLab.src.models.projection import My_proj
from transformers import CLIPVisionModelWithProjection
from src.DepthLab.inference.depthlab_pipeline import DepthLabPipeline
from src.DepthLab.utils.seed_all import seed_all
from src.DepthLab.utils.image_util import get_filled_for_latents

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from scipy.ndimage import zoom


def upsample_dense_depth_scipy(depth_map, target_shape, order=1):
    """
    使用Scipy的多阶插值进行上采样
    参数：
        depth_map: 输入稠密深度图，形状为[h1, w1]
        target_shape: 目标形状，元组(h2, w2)
        order: 插值阶数 (0:最近邻, 1:双线性, 3:双三次)
    返回：
        上采样后的深度图，形状为[h2, w2]
    """
    h1, w1 = depth_map.shape
    h2, w2 = target_shape

    # 计算缩放因子
    zoom_factor = (h2 / h1, w2 / w1)

    # 使用scipy的zoom函数插值
    upsampled = zoom(depth_map, zoom_factor, order=order)

    return upsampled


def downsample_sparse_depth(depth_map, target_shape, invalid_val=0):
    """
    使用最近邻插值法降采样稀疏深度图，保留最近的有效值。
    参数：
        depth_map: 输入的稀疏深度图，形状为[h1, w1]
        target_shape: 目标形状，元组(h2, w2)
        invalid_val: 无效值的标识，默认为0
    返回：
        降采样后的稀疏深度图，形状为[h2, w2]
    """
    h1, w1 = depth_map.shape
    h2, w2 = target_shape

    # 提取有效点坐标和值
    valid_mask = depth_map != invalid_val
    rows, cols = np.where(valid_mask)
    values = depth_map[rows, cols]

    if len(values) == 0:
        return np.full(target_shape, invalid_val)

    # 将原图坐标映射到目标图空间（基于像素中心）
    src_y = (rows + 0.5) / h1 * h2
    src_x = (cols + 0.5) / w1 * w2
    points = np.column_stack((src_y, src_x))

    # 构建最近邻插值器
    interpolator = NearestNDInterpolator(points, values)

    # 生成目标图像素中心的查询坐标
    query_y, query_x = np.mgrid[0.5:h2, 0.5:w2]
    query_points = np.column_stack((query_y.ravel(), query_x.ravel()))

    # 插值并处理无效区域
    downsampled = interpolator(query_points).reshape(h2, w2)
    downsampled = np.nan_to_num(downsampled, nan=invalid_val)

    return downsampled


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["H"]
        self.width = dataset_config["W"]
        self.fx = dataset_config["fx"]
        self.fy = dataset_config["fy"]
        self.cx = dataset_config["cx"]
        self.cy = dataset_config["cy"]

        self.depth_scale = dataset_config["depth_scale"]
        self.distortion = np.array(
            dataset_config['distortion']) if 'distortion' in dataset_config else None
        self.crop_edge = dataset_config['crop_edge'] if 'crop_edge' in dataset_config else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32)

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose @ c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "rgb").glob("*.png")), key=lambda x: int(os.path.basename(x)[-9:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.TIFF")), key=lambda x: int(os.path.basename(x)[-10:-5]))
        self.n_img = len(self.color_paths)
        self.load_poses(self.dataset_path / "gt_pose.txt")

    def load_poses(self, path):
        self.poses = []
        pose_data = np.loadtxt(path, delimiter=" ", dtype=np.unicode_, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        for i in range(self.n_img):
            quat = pose_vecs[i][4:]
            trans = pose_vecs[i][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            pose = T
            self.poses.append(pose)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index]


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):

        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index]


class depth_cfg():
    def __init__(self):
        self.denoise_steps = 50
        self.processing_res = 768
        self.normalize_scale = 1
        self.strength = 0.8
        self.seed = 1234
        self.pretrained_model_name_or_path = './src/DepthLab/checkpoints/marigold-depth-v1-0'
        self.image_encoder_path = './src/DepthLab/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K'
        self.denoising_unet_path = './src/DepthLab/checkpoints/DepthLab/denoising_unet.pth'
        self.mapping_path = './src/DepthLab/checkpoints/DepthLab/mapping_layer.pth'
        self.reference_unet_path = './src/DepthLab/checkpoints/DepthLab/reference_unet.pth'
        self.blend = True
        self.refine = False


class Kitti(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.load_dir = "/data1/KITTI/Kitti_raw"
        self.save_path = "./output/kitti"
        self.color_paths = []
        self.depth_paths = []
        self.points_path = []
        self.poses = []
        self.load_data()
        self.args = depth_cfg()

        vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path,
                                            subfolder='vae')
        text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
        denoising_unet = UNet2DConditionModel_main.from_pretrained(self.args.pretrained_model_name_or_path,
                                                                   subfolder="unet",
                                                                   in_channels=12, sample_size=96,
                                                                   low_cpu_mem_usage=False,
                                                                   ignore_mismatched_sizes=True)
        reference_unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet",
                                                              in_channels=4, sample_size=96,
                                                              low_cpu_mem_usage=False,
                                                              ignore_mismatched_sizes=True)
        image_enc = CLIPVisionModelWithProjection.from_pretrained(self.args.image_encoder_path)
        mapping_layer = My_proj()

        mapping_layer.load_state_dict(
            torch.load(self.args.mapping_path, map_location="cpu"),
            strict=False,
        )
        mapping_device = torch.device("cuda")
        mapping_layer.to(mapping_device)
        reference_unet.load_state_dict(
            torch.load(self.args.reference_unet_path, map_location="cpu"),
        )
        denoising_unet.load_state_dict(
            torch.load(self.args.denoising_unet_path, map_location="cpu"),
            strict=False,
        )
        tokenizer = CLIPTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder='tokenizer')
        scheduler = DDIMScheduler.from_pretrained(self.args.pretrained_model_name_or_path, subfolder='scheduler')
        self.pipe = DepthLabPipeline(reference_unet=reference_unet,
                                     denoising_unet=denoising_unet,
                                     mapping_layer=mapping_layer,
                                     vae=vae,
                                     text_encoder=text_encoder,
                                     tokenizer=tokenizer,
                                     image_enc=image_enc,
                                     scheduler=scheduler,
                                     ).to('cuda')

    def get_kitti_data_raw(self, basedir):
        date = basedir.split("/")[-2]
        drive = basedir.split("/")[-1].split('_')[-2]
        data = pykitti.raw(self.load_dir, date, drive)
        return data

    def get_kitti_data_odometry(self, basedir):
        sequence = '00'
        data = pykitti.odometry(basedir, sequence)
        return data

    def load_data(self):

        basedir = "/home/kitt_odometry/dataset/"

        kitti_data = self.get_kitti_data_odometry(basedir)
        cam2_Ks = kitti_data.calib.K_cam2
        cam3_Ks = kitti_data.calib.K_cam3

        self.K = cam2_Ks

        self.fx = cam2_Ks[0, 0]
        self.fy = cam2_Ks[1, 1]
        self.cx = cam2_Ks[0, 2]
        self.cy = cam2_Ks[1, 2]

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))

        T_cam2_velo = kitti_data.calib.T_cam2_velo
        print("T", T_cam2_velo)
        T_cam0_velo = kitti_data.calib.T_cam0_velo
        self.cam2_to_velo = T_cam2_velo

        image = Image.open(kitti_data.cam2_files[0])
        width, height = image.size
        self.width = width
        self.height = height

        for frame_idx in range(len(kitti_data)):
            # Points are already in ego frame (velodyne frame), so we don't need to transform them
            self.color_paths.append(kitti_data.cam2_files[frame_idx])
            self.points_path.append(kitti_data.velo_files[frame_idx])
            T_cam0_w = kitti_data.poses[frame_idx]
            T_velo_cam0 = np.linalg.inv(T_cam0_velo)
            self.velo2world = T_cam0_w @ T_velo_cam0
            velo_to_world_current = self.velo2world
            # if frame_idx == 0:
            #     velo_to_world_start = self.velo2world
            # # Compute ego_to_world transformation relative to the start
            # ego_to_world = np.linalg.inv(velo_to_world_start) @ velo_to_world_current

            # Compute camera-to-world transformation
            cam2world = velo_to_world_current @ T_cam2_velo

            # No need to apply OPENCV2DATASET as KITTI uses the same coordinate system

            # self.poses.append(cam2world)
            self.poses.append(cam2world)

        # for frame_idx in range(len(kitti_data)):
        #     # Points are already in ego frame (velodyne frame), so we don't need to transform them
        #     self.color_paths.append(kitti_data.cam2_files[frame_idx])
        #     self.points_path.append(kitti_data.velo_files[frame_idx])
        #     T_cam0_w = kitti_data.poses[frame_idx]
        #     T_velo_cam0 = np.linalg.inv(T_cam0_velo)
        #     self.velo2world = T_cam0_w @ T_velo_cam0
        #     velo_to_world_current = self.velo2world
        #     if frame_idx==0:
        #         velo_to_world_start = self.velo2world
        #     # Compute ego_to_world transformation relative to the start
        #     ego_to_world = np.linalg.inv(velo_to_world_start) @ velo_to_world_current
        #
        #      # Compute camera-to-world transformation
        #     cam2world = velo_to_world_current #@ T_cam2_velo
        #
        #     # No need to apply OPENCV2DATASET as KITTI uses the same coordinate system
        #
        #
        #     self.poses.append(cam2world)
        #
        # self.poses.append(T_cam0_w)

    def __getitem__(self, index):

        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)
        # color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        c2w = self.poses[index]
        point = np.fromfile(self.points_path[index], dtype=np.float32).reshape(-1, 4)
        point_xyz = point[:, :3]
        point_camera = (np.pad(point_xyz, ((0, 0), (0, 1)), constant_values=1) @ self.cam2_to_velo.T)[:, :3]

        pts_depth = np.zeros([1, self.height, self.width])
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        uvz = point_camera[point_camera[:, 2] > 0]
        uvz = uvz @ self.K.T
        uvz[:, :2] /= uvz[:, 2:]
        uvz = uvz[uvz[:, 1] >= 0]
        uvz = uvz[uvz[:, 1] < self.height]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < self.width]
        uv = uvz[:, :2]
        uv = uv.astype(int)

        # pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]
        # # pts_depth = torch.from_numpy(pts_depth).float()
        # depth = np.array(pts_depth.squeeze(0))
        #
        # print("----------------------")
        # print(type(depth), depth.shape)
        # input_image = Image.open(self.color_paths[index])
        # input_image = input_image.resize((self.width, self.height))
        # depth_numpy = depth
        # mask = (depth_numpy == 0).astype(np.uint8)

        # print("earewr",self.width*self.height - mask.sum())
        # print('-----------------')
        # print("img", input_image.size)
        # print("depth", depth_numpy.shape,depth_numpy.max(),depth_numpy.min())
        # print("mask", mask.shape)
        #
        # x, y = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32), indexing='xy') # pixels
        # grid = np.stack((x,y), axis=-1).reshape(-1,2)
        #
        # depthj = interp_grid((uv[:, 0], uv[:, 1]), uvz[:, 2], grid, method='nearest', fill_value=0).reshape(self.height,self.width)
        # depthj[140:170,:] = 0
        # mask = (depthj == 0).astype(np.uint8)

        max_depth = np.max(uvz[:, 2])
        min_depth = np.min(uvz[:, 2])
        values = uvz[:, 2].astype(float)
        print("max", max_depth, min_depth)
        depth[uv[:, 1], uv[:, 0]] = values

        print("----------------------")

        input_image = Image.open(self.color_paths[index])
        # input_image = input_image.resize((self.width, self.height))
        depth_numpy = depth
        mask = (depth_numpy == 0).astype(np.uint8)
        print('-----------------')
        print("img", input_image.size)
        print("depth", depth_numpy.shape, depth_numpy.max(), depth_numpy.min())
        print("mask", mask.shape)

        cv2.imwrite("depth_norm.jpg", depth)
        cv2.imwrite("mask.jpg", mask)
        # np.save("depth_norm.npy", depth)
        # np.save("mask.npy", mask)
        if self.args.refine is not True:
            depth_numpy = get_filled_for_latents(mask, depth_numpy)
        output_depth_color_pth = os.path.join("/home/LGS-SLAM/output/kitti/00_0502", "depth_colored")
        os.makedirs(output_depth_color_pth, exist_ok=True)

        pipe_out = self.pipe(
            input_image,
            denosing_steps=self.args.denoise_steps,
            processing_res=self.args.processing_res,
            match_input_res=True,
            batch_size=1,
            color_map="Spectral",
            show_progress_bar=True,
            depth_numpy_origin=depth_numpy,
            mask_origin=mask,
            guidance_scale=1,
            normalize_scale=self.args.normalize_scale,
            strength=self.args.strength,
            blend=self.args.blend)

        depth_pred: np.ndarray = pipe_out.depth_np
        print(depth_pred.shape, type(depth_pred), depth_pred.max(), depth_pred.min())
        output_depth_color = "/home/LGS-SLAM/output/kitti/00_0502" + "/depth_colored/" + str(index) + ".png"
        pipe_out.depth_colored.save(output_depth_color)

        # Depth = (depthj - depthj.min()) / (depthj.max() - depthj.min()) * 255.0
        # Depth = Depth.astype(np.uint8)
        # Depth = np.repeat(Depth[..., np.newaxis], 3, axis=-1)
        # image_dir = os.path.join(self.save_path, "rgb_image_2")
        # mkdir_p(image_dir)
        # image_path = os.path.join(self.save_path, "rgb_image_2", str(index).zfill(7) + '.png')
        # cv2.imwrite(image_path, color_data)
        # depth_dir = os.path.join(self.save_path, "depth_image_0")
        # mkdir_p(depth_dir)
        # depth_path = os.path.join(self.save_path, "depth_image_0", str(index).zfill(7) + '.png')
        # cv2.imwrite(depth_path, depthj)

        return index, color_data, depth_pred, self.poses[index], point_camera, depth_pred  # depthj


def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == "Kitti":
        return Kitti
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
