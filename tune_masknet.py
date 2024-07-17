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
import wandb
from eval_metric import calculate_metrics
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
from scene.lightweight_seg import Context_Guided_Network
from torchvision.models.detection import maskrcnn_resnet50_fpn
def training():
  with wandb.init() as run:
    config=wandb.config
    args.model_path_args=args.model_path_args_+"_"+str(run.name)
        # wandb.run.log_code(".")
    
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from=opt_g
    dataset.kernel_size=config.kernel_size
    first_iter = 0
    scale=2

    implicit_mask = Context_Guided_Network(classes= 1, M= 2, N= 2, input_channel=3).cuda()
    maskrcnn=maskrcnn_resnet50_fpn(pretrained=True).cuda()
    maskrcnn.eval()
    for param in maskrcnn.parameters():##cnn after avgpool should train?
          param.requires_grad = False
    l=[]
    l.append({'params': implicit_mask.parameters(), 'lr': 5e-4, "name": "segment_net"})
    optimizer= torch.optim.Adam(l, eps=1e-15)
    tb_writer = prepare_output_and_logger(dataset)
    scene = Scene(dataset, gaussians=None, resolution_scales=[scale])
    # gaussians.training_setup(opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras(scale).copy()
    testCameras = scene.getTestCameras(scale).copy()
    allCameras = trainCameras + testCameras
    ##initilize masknet
    # highresolution index
    viewpoint_stack = scene.getTrainCameras(scale).copy()
    len_cam=len(viewpoint_stack)
    embedding_a_list = [None] * (len(viewpoint_stack)+1)
    print("saving all gt_app_embdeeing to embedding list")
    # networks=(gaussians.feature_linear, gaussians.app_encoder, gaussians.style_transfer,  gaussians.decoder)
    # training_report(tb_writer, 0, 0, 0, 0, 9, [0], scene, render, (pipe, background, dataset.kernel_size), scale, networks=networks)

    # for viewpoint_cam in viewpoint_stack:

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        


        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras(scale).copy() +  scene.getTestCameras(scale).copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        gt_image=viewpoint_cam.original_image.to("cuda")
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
        
        gt_image=gt_image.unsqueeze(0)
        pred_mask_im=implicit_mask(gt_image)

        with torch.no_grad(): 
                        
                        prediction=maskrcnn(gt_image)
                        mask = prediction[0]['masks']  #[1:2,...]
                        labels=prediction[0]['labels']
                        idx=torch.where(labels==1)
                        mask=mask[idx]
                        if mask.shape[0]==0:
                            pred_mask_mc=torch.zeros_like(gt_image)[:,:1,...]
                        else:pred_mask_mc, _ = torch.max(mask, dim=0, keepdim=True)
        
        

        

        loss_dict={}
        Ll1 = l1_loss(pred_mask_mc, pred_mask_im)
        loss = Ll1 
        loss_dict["Ll1"]=Ll1.item()
        
        loss_dict["loss"]=loss.item()
        
        loss.backward()

        iter_end.record()
        if iteration % 1000 == 0:
                    with torch.no_grad():
                        gt_image_im=gt_image*(1-pred_mask_im)
                        gt_image_mc=gt_image*(1-pred_mask_mc)
                        gt_img = gt_image.squeeze(0)
                        log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        log_mask_im = ((gt_image_im.squeeze(0)).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        log_mask_mc = ((gt_image_mc.squeeze(0)).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        
                        log_img = np.concatenate([log_gt_img, log_mask_mc, log_mask_im ], axis=1)
                        
                        save_images(log_img, f'{args.model_path_args}/train_images/{iteration:06d}.png')
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                for k,v in loss_dict.items():
                        if args.wandb:wandb.log({k: v}, step=iteration)
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()



            
            # Optimizer step
            if iteration < opt.iterations:
                optimizer.step()
                optimizer.zero_grad(set_to_none = True)
            if iteration == opt.iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((implicit_mask.state_dict(), iteration), args.model_path_args + "/implicit_net" + str(iteration) + ".pth")
                

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
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras(scale)}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras(scale)[idx % len(scene.getTrainCameras(scale))] for idx in range(5, 30, 5)]})

        for config_ in validation_configs:
            if config_['cameras'] and len(config_['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config_['cameras']):
                    gt_image=viewpoint.original_image.to("cuda")
                    gt_image_features_ = app_encoder(normalize_vgg(gt_image.unsqueeze(0)))
                    gt_image_features=gt_image_features_.relu3_1

                    

                    rendered_feature = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    
                    tranfered_features = style_transfer(
                        rendered_feature.unsqueeze(0),#.detach(), # point cloud features [N, C]
                        gt_image_features,
                    )
                    image = decoder(tranfered_features)
                    
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)
                    if config_["name"]=="test":
                        img_pred = image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                        img_pred_ = (img_pred*255).astype(np.uint8)
                        gt_image_np = gt_image.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                        gt_image_np = (gt_image_np*255).astype(np.uint8)
                        imageio.imwrite(os.path.join(args.model_path_args, "pred", f'{idx:03d}.png'), img_pred_)
                        imageio.imwrite(os.path.join(args.model_path_args, "gt", f'{idx:03d}.png'), gt_image_np)

                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config_['cameras'])
                l1_test /= len(config_['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config_['name'], l1_test, psnr_test))
        if args.wandb:
            wandb.log({'total_points': scene.gaussians.get_xyz.shape[0]}, iteration)
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
    parser.add_argument("--model_path_args_", type=str, default="output/test1")
    parser.add_argument('--encode_a_random', action='store_true', default=True)
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument("--iterations_", type=int, default=30000)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    opt = op.extract(args)
    opt.iterations = args.iterations_
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    

    if args.wandb:
        # wandb.config = {"scale": args.scale, "exp_name": args.exp_name, "dataset": args.source_path.split("/")[-1],"savepath":args.model_path, "feature_linear_lr":1e-3, "decoder_lr":1e-3, "style_trans_lr":1e-3, "app_encoder_lr":1e-3,"segnet_lr":1e-3}
        
        sweep_config = {'method': 'random'}
        metric = {
            'name': 'loss',
            'goal': 'minimize'   
            }

        sweep_config['metric'] = metric
        parameters_dict = {
            'decoder_lr': {
                'values': [5e-4]
                },
            'style_trans_lr': {
                'values': [1e-4]
                },
            'app_encoder_lr': {
                'values': [5e-4] #1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
                },
            'segnet_lr':{
                'values': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] 
                }, #1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
            'densify_until_iter':{
                'values': [15000]
                },
            'densify_from_iter':{
                'values': [3000]
                },
            'densification_interval':{
                'values': [3000]
                },
            'maskrs_max':{
                'values': [200]
                },
            # 'prune_only_interval':{
            #     'values': [3000,1000]
            #     },
            # 'prune_only_start_step':{
            #     'values': [1000,5000]
            #     },
            # 'prune_only_end_step':{
            #     'values': [30000,15000,10000]
            #     },
            'prune_size_threshold':{
                'values': [0.2]
                },
            "apploss_random_ratio":{
                'values': [0.1]
            },
            "apploss_pair_ratio":{
                'values': [0.5]
            },
            "lambda_dssim":{
                'values': [0.55]
            },
            "kernel_size":{
                "values":[0.01]
            }
            }
            
        sweep_config['parameters'] = parameters_dict


        global opt_g
        opt_g = (lp.extract(args), opt, pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
        sweep_id = wandb.sweep(sweep_config, project="gaussian-mip")
                # Initialize system state (RNG)
        safe_state(args.quiet)

        # Start GUI server, configure and run training
        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        # training(lp.extract(args), opt, pp.extract(args),args)
        
        # All done
        
        print("Optimizing " + args.model_path_args)
        if os.path.exists(os.path.join(args.model_path_args, "pred")) and args.exp_name!="default":
            print("Model path already exists, exiting")
            exit(1)
        print("\nReconstruction complete.")
        wandb.agent(sweep_id, training, count=6)
    else:
        training()
        # All done
        print("\nTraining complete.")
