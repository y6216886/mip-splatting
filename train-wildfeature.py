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
import numpy as np
import open3d as o3d
import cv2
import torch
import torch.nn.functional as F
from utils.image_utils import save_images
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from scene.VGG import VGGEncoder, normalize_vgg
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    scale=4
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    # gaussians.training_setup(opt)
    vgg_encoder = VGGEncoder().cuda()
    gaussians.training_setup_style_feature(opt, vgg_encoder, args)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras(scale).copy()
    testCameras = scene.getTestCameras(scale).copy()
    allCameras = trainCameras + testCameras
    
    # highresolution index
    viewpoint_stack = scene.getTrainCameras(scale).copy()
    len_cam=len(viewpoint_stack)
    embedding_a_list = [None] * (len(viewpoint_stack)+1)
    print("saving all gt_app_embdeeing to embedding list")
    for viewpoint_cam in viewpoint_stack:
                id=viewpoint_cam.uid
                print("uid",id)
                gt_image = viewpoint_cam.original_image.unsqueeze(0).cuda()
                gt_image_features=gaussians.app_encoder(normalize_vgg(gt_image))
                embedding_a_list[id]=gt_image_features.relu3_1
    print("saving are done")
    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        # if random.random() < 0.3 and dataset.sample_more_highres:
            # viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
        # viewpoint_cam = trainCameras[scale]
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        rendered_feature, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        id=viewpoint_cam.uid
        gt_image_features_=embedding_a_list[id]
        gt_image_features=gt_image_features_.detach()
        tranfered_features = gaussians.style_transfer(
                rendered_feature.unsqueeze(0), #.detach(), # point cloud features [N, C]
                gt_image_features, 
            )
        image = gaussians.decoder(tranfered_features)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # sample gt_image with subpixel offset
        if args.mask:
                if args.masktype=="context": pred_mask=gaussians.implicit_mask(gt_image.unsqueeze(0))
                elif args.masktype=="maskrcnn":
                    with torch.no_grad(): 
                        gt_image=gt_image.unsqueeze(0)
                        prediction=gaussians.implicit_mask(gt_image)
                        mask = prediction[0]['masks']  #[1:2,...]
                        labels=prediction[0]['labels']
                        idx=torch.where(labels==1)
                        mask=mask[idx]
                        if mask.shape[0]==0:
                            pred_mask=torch.zeros_like(gt_image)[:,:1,...]
                        else:pred_mask, _ = torch.max(mask, dim=0, keepdim=True)
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
        if args.mask:
            image_=image*(1-pred_mask)
            gt_image_=gt_image*(1-pred_mask)
        else:
            image_=image
            gt_image_=gt_image
        Ll1 = l1_loss(image_, gt_image_)
        loss = (1.0 - opt.lambda_dssim)* Ll1 +  opt.lambda_dssim*(1.0 - ssim(image_, gt_image_))
        apploss=0
        if args.appearance:
                # breakpoint()
                pred_app_feature = gaussians.app_encoder(normalize_vgg(image))
                # pred_app_feature=pred_app_feature.relu3_1
                # apploss+=F.mse_loss(pred_app_feature.relu1_1,gt_image_features_.relu1_1) 
                # apploss+=F.mse_loss(pred_app_feature.relu2_1,gt_image_features_.relu2_1) 
                apploss+=F.mse_loss(pred_app_feature.relu3_1,gt_image_features) 
                # apploss+=F.mse_loss(pred_app_feature.relu4_1,gt_image_features_.relu4_1) 
                # regu_app=_l2_regularize(pred_app_feature)
        if args.encode_a_random:

                idx_random=random.choice(range(len_cam))
                appearance_embeding_random_=embedding_a_list[idx_random]
                appearance_embeding_random=appearance_embeding_random_.detach()
                tranfered_features_random = gaussians.style_transfer(
                rendered_feature.unsqueeze(0),#.detach(), # point cloud features [N, C]
                appearance_embeding_random,
            )
                image_random = gaussians.decoder(tranfered_features_random)
                pred_app_feature_random = gaussians.app_encoder(normalize_vgg(image_random))
                apploss+=F.mse_loss(pred_app_feature_random.relu3_1, appearance_embeding_random)
        loss += apploss*1 #+ regu_app*config.app_reg
        # loss_dict["apploss"]=apploss.item()
                # loss_dict["regu_app"]=regu_app.item()

        
        loss.backward()

        iter_end.record()
        if iteration % 1000 == 0:
                    with torch.no_grad():
                        pred_img = image.squeeze(0)
                        gt_img = gt_image.squeeze(0)
                        log_pred_img = (pred_img.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        if args.mask:
                            log_mask = ((pred_mask.squeeze(0).repeat(3,1,1)).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            log_img = np.concatenate([log_gt_img, log_pred_img, log_mask], axis=1)
                        
                        else:
                            log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                        save_images(log_img, f'{args.model_path_args}/train_images/{iteration:06d}.png')
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            networks=(gaussians.feature_linear, gaussians.app_encoder, gaussians.style_transfer,  gaussians.decoder)
            
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size), scale, networks=networks)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
#     if tb_writer:
#         tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
#         tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
#         tb_writer.add_scalar('iter_time', elapsed, iteration)

#     # Report test and samples of training set
#     if iteration in testing_iterations:
#         torch.cuda.empty_cache()
#         validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
#                               {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

#         for config in validation_configs:
#             if config['cameras'] and len(config['cameras']) > 0:
#                 l1_test = 0.0
#                 psnr_test = 0.0
#                 for idx, viewpoint in enumerate(config['cameras']):
#                     image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
#                     gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
#                     if tb_writer and (idx < 5):
#                         tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                         if iteration == testing_iterations[0]:
#                             tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
#                     l1_test += l1_loss(image, gt_image).mean().double()
#                     psnr_test += psnr(image, gt_image).mean().double()
#                 psnr_test /= len(config['cameras'])
#                 l1_test /= len(config['cameras'])          
#                 print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
#                 if tb_writer:
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
#                     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

#         if tb_writer:
#             tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
#             tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
#         torch.cuda.empty_cache()


import imageio
def training_report(wandb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, scale, networks):
    feature_linear, app_encoder, style_transfer, decoder=networks
    if not os.path.exists(os.path.join(args.model_path_args, "pred")):
        os.makedirs(os.path.join(args.model_path_args, "pred"))
    if not os.path.exists(os.path.join(args.model_path_args, "gt")):
        os.makedirs(os.path.join(args.model_path_args, "gt"))
        # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras(scale)[idx % len(scene.getTrainCameras(scale))] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    gt_image=viewpoint.original_image.to("cuda")
                    gt_image_features_ = app_encoder(normalize_vgg(gt_image.unsqueeze(0)))
                    gt_image_features=gt_image_features_.relu3_1
                    # final_vgg_features=gaussians.feature_linear.forward_directly_on_point(gaussians._vgg_features)

                    

                    rendered_feature = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    
                    tranfered_features = style_transfer(
                        rendered_feature.unsqueeze(0),#.detach(), # point cloud features [N, C]
                        gt_image_features,
                    )
                    image = decoder(tranfered_features)
                    
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)
                    if config["name"]=="test":
                        img_pred = image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                        img_pred_ = (img_pred*255).astype(np.uint8)
                        gt_image_np = gt_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                        gt_image_np = (gt_image_np*255).astype(np.uint8)
                        imageio.imwrite(os.path.join(args.model_path_args, "pred", f'{idx:03d}.png'), img_pred_)
                        imageio.imwrite(os.path.join(args.model_path_args, "gt", f'{idx:03d}.png'), gt_image_np)
                    # if tb_writer and (idx < 5):
                    #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    #     if iteration == testing_iterations[0]:
                    #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    #save image and gt_image
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

        torch.cuda.empty_cache()
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--masktype', type=str, default="maskrcnn", #maskrcnn context
                        help='mode seeking')
    parser.add_argument('--mask', action='store_true', default=False)
    parser.add_argument('--appearance', action='store_true', default=False)
    parser.add_argument("--model_path_args", type=str, default="output/test1")
    parser.add_argument('--encode_a_random', action='store_true', default=True)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if os.path.exists(os.path.join(args.model_path_args, "pred")) and args.exp_name!="default":
        print("Model path already exists, exiting")
        exit(1)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")