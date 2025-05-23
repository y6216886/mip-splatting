#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import copy
import pandas as pd

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

import cv2

# def readColmapCameras_fast3r(cam_extrinsics, cam_intrinsics, images_folder):
#     cam_infos = []
#     poses=[]
#     for idx, key in enumerate(cam_extrinsics):
#         sys.stdout.write('\r')
#         # the exact output you're looking for:
#         sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
#         sys.stdout.flush()

#         extr = cam_extrinsics[key]
#         intr = cam_intrinsics[extr.camera_id]
#         height = intr.height
#         width = intr.width

#         uid = intr.id
#         R = np.transpose(qvec2rotmat(extr.qvec))
#         T = np.array(extr.tvec)
#         pose = np.block([[R, T.reshape(3, 1)], [np.zeros((1, 3)), 1]])
#         poses.append(pose)

#         image_path = os.path.join(images_folder, os.path.basename(extr.name))
#         image_name = os.path.basename(image_path).split(".")[0]
#         image = Image.open(image_path)

#         if intr.model=="SIMPLE_PINHOLE":
#             focal_length_x = intr.params[0]
#             FovY = focal2fov(focal_length_x, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model=="PINHOLE":
#             focal_length_x = intr.params[0]
#             focal_length_y = intr.params[1]
#             FovY = focal2fov(focal_length_y, height)
#             FovX = focal2fov(focal_length_x, width)
#         elif intr.model=="SIMPLE_RADIAL":
#             f, cx, cy, r = intr.params
#             FovY = focal2fov(f, height)
#             FovX = focal2fov(f, width)
#             prcppoint = np.array([cx / width, cy / height])
#             # undistortion
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
#             D = np.array([r, 0, 0, 0])  # Only radial distortion
#             image_undistorted = cv2.undistort(image_cv, K, D, None)
#             image_undistorted = cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2RGB)
#             image = Image.fromarray(image_undistorted)
#         else:
#             assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

#         cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
#                               image_path=image_path, image_name=image_name, width=width, height=height)
#         cam_infos.append(cam_info)
#     sys.stdout.write('\n')
#     return cam_infos, poses


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" :
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=='SIMPLE_RADIAL':
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # get rid of too many opened files
        image = copy.deepcopy(image)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readColmapCameras_vgtt(cam_extrinsics, cam_intrinsics, images_folder, imgsize=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        if imgsize is not None:
            
            new_width = imgsize
            height = round(height * (new_width / width) / 14) * 14
            width = new_width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # get rid of too many opened files
        image = copy.deepcopy(image)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    df = pd.read_csv("/U_20240109_SZR_SMIL/yyf/young/code/stylegs/data/split/brandenburg_gate.tsv", sep="\t")
    train_file_name=df[df['split']=='train']['filename'].tolist()
    test_file_name=df[df['split']=='test']['filename'].tolist()
    if eval:
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name+'.jpg' in train_file_name]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name+'.jpg' in test_file_name]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo_vgtt(path, images, eval, llffhold=8):
    # subpath="sparse_200_518"
    print("note the subpath!!")
    imgsize=518
    subpath="vgtt"
    try:
        cameras_extrinsic_file = os.path.join(path, f"sparse/{subpath}", "images.bin")
        cameras_intrinsic_file = os.path.join(path, f"sparse/{subpath}", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, f"sparse/{subpath}", "images.txt")
        cameras_intrinsic_file = os.path.join(path, f"sparse/{subpath}", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras_vgtt(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), imgsize=imgsize)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if os.path.basename(path) in ["brandenburg_gate"]:
        df = pd.read_csv("/U_20240109_SZR_SMIL/yyf/young/code/stylegs/data/split/brandenburg_gate.tsv", sep="\t")
        train_file_name=df[df['split']=='train']['filename'].tolist()
        test_file_name=df[df['split']=='test']['filename'].tolist()
        # train_file_name=train_file_name[:100]
        # test_file_name=test_file_name[:2]
        if eval:
            # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name+'.jpg' in train_file_name]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if c.image_name+'.jpg' in test_file_name]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
    else:
        train_cam_infos = [c for idx, c in enumerate(cam_infos)][:-3]
        test_cam_infos = [c for idx, c in enumerate(cam_infos)][-3:]
    #get the intersection of the 

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, f"sparse/{subpath}/points3D.ply")
    bin_path = os.path.join(path, f"sparse/{subpath}/points3D.bin")
    txt_path = os.path.join(path, f"sparse/{subpath}/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# def readColmapSceneInfo_fast3r(path, images, eval, llffhold=8):

#     # if eval:
#     #     cameras_extrinsic_file = os.path.join(path, f"sparse/1", "images.txt")
#     #     cameras_intrinsic_file = os.path.join(path, f"sparse/1", "cameras.txt")
#     # else:
#     cameras_extrinsic_file = os.path.join(path, f"sparse/0", "images.txt")
#     cameras_intrinsic_file = os.path.join(path, f"sparse/0", "cameras.txt")

#     cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
#     cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
#     reading_dir = "images" if images == None else images

#     cam_infos_unsorted, poses = readColmapCameras_fast3r(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
#     sorting_indices = sorted(range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name)
#     cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
#     sorted_poses = [poses[i] for i in sorting_indices]
#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

#     if eval:
#         train_cam_infos = cam_infos
#         test_cam_infos = cam_infos
#         train_poses = sorted_poses
#         test_poses = sorted_poses
#     else:
#         train_cam_infos = cam_infos
#         test_cam_infos = []
#         train_poses = sorted_poses
#         test_poses = []

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     ply_path = os.path.join(path, f"sparse/0/points3D.ply")
#     bin_path = os.path.join(path, f"sparse/0/points3D.bin")
#     txt_path = os.path.join(path, f"sparse/0/points3D.txt")
#     if not os.path.exists(ply_path):
#         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
#         try:
#             xyz, rgb, _ = read_points3D_binary(bin_path)
#         except:
#             xyz, rgb, _ = read_points3D_text(txt_path)
#         storePly(ply_path, xyz, rgb)
#     try:
#         pcd = fetchPly(ply_path)
#     except:
#         pcd = None

#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path,
#                            train_poses=train_poses,
#                            test_poses=test_poses)
#     return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMultiScale(path, white_background,split, only_highres=False):
    cam_infos = []
    
    print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]
        
    meta = {k: np.array(meta[k]) for k in meta}
    
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        fovy = focal2fov(meta["focal"][idx], image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres=False):
    print("Reading train from metadata.json")
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    print("number of training images:", len(train_cam_infos))
    print("Reading test from metadata.json")
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    print("number of testing images:", len(test_cam_infos))
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "vgtt": readColmapSceneInfo_vgtt,
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
}
