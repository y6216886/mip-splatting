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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state,PILtoTorch
from argparse import ArgumentParser,Namespace
# from arguments import ModelParams, PipelineParams, get_combined_args
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import GaussianModel
import copy,pickle,time
from utils.general_utils import *
import imageio
#


def render_interpolate(model_path, name, iteration, views, gaussians, pipeline, background,select_idxs=None):
    if args.scene_name=="brandenburg":
        select_idxs=[88]#
    elif args.scene_name=="sacre":
        select_idxs=[29]
    elif args.scene_name=="trevi":
        select_idxs=[55]
        
    render_path = os.path.join(model_path, name,"ours_{}".format(iteration), f"intrinsic_dynamic_interpolate")
    render_path_gt = os.path.join(model_path, name,"ours_{}".format(iteration), f"intrinsic_dynamic_interpolate","refer")
    makedirs(render_path, exist_ok=True)
    makedirs(render_path_gt, exist_ok=True)
    inter_weights=[i*0.1 for i in range(0,21)]
    select_views=[views[i] for i in select_idxs]
    for idx, view in enumerate(tqdm(select_views, desc="Rendering progress")):
        
        torchvision.utils.save_image(view.original_image, os.path.join(render_path_gt, f"{select_idxs[idx]}_{view.colmap_id}" + ".png"))
        sub_s2d_inter_path=os.path.join(render_path,f"{select_idxs[idx]}_{view.colmap_id}")
        makedirs(sub_s2d_inter_path, exist_ok=True)
        for inter_weight in  inter_weights:
            gaussians.colornet_inter_weight=inter_weight
            rendering = render(view, gaussians, pipeline, background)["render"]
            torchvision.utils.save_image(rendering, os.path.join(sub_s2d_inter_path, f"{idx}_{inter_weight:.2f}" + ".png"))
    gaussians.colornet_inter_weight=1.0


def render_multiview_vedio(model_path, name, train_views, test_views, gaussians, pipeline, background,args,dataset):
    
    if args.scene_name=="brandenburg":
        format_idx=11#4
        # select_view_id=[12, 59, 305]
        select_view_id=[120, 159, 505]
        length_view=90*2
        appear_idxs=[313,78]#
        name="train"
        view_appears=[train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[0,1,2,3,4,5,7,8,9]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views=[train_views[i] for i in select_view_id]
    elif args.scene_name=="sacre":
        format_idx=38 #
        select_view_id=[753,657,595,181,699,]#700
        length_view=45*2
        
        appear_idxs=[350,76]
        name="train"
        view_appears=[train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[6,12,15,17]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views=[train_views[i] for i in select_view_id]
    elif args.scene_name=="trevi":
        format_idx=17 
        select_view_id=[408,303,79,893,395,281]#700
        length_view=45*2
        
        appear_idxs=[317,495]
        
        name="train"
        view_appears=[train_views[i] for i in appear_idxs]

        # intrinsic_idxs=[0,2,3,8,9,11]
        # name="test"
        # view_intrinsics=[test_views[i] for i in intrinsic_idxs]
        views=[train_views[i] for i in select_view_id]
        
    for vid, view_appear in enumerate(tqdm(view_appears, desc="Rendering progress")):
        view_appear.image_height,view_appear.image_width=train_views[format_idx].image_height,train_views[format_idx].image_width
        view_appear.FoVx,view_appear.FoVy=train_views[format_idx].FoVx,train_views[format_idx].FoVy
        appear_idx=appear_idxs[vid]
        generated_views=generate_multi_views(views, view_appear,length=length_view)
        render_path = os.path.join(os.path.dirname(args.ckpt_path),"demos" ,f"multiview_vedio",f"{name}_{appear_idx}_{view_appear.colmap_id}") #
        makedirs(render_path, exist_ok=True)
        
        render_video_out = imageio.get_writer(f'{render_path}/000_mv_{name}_{appear_idx}_{view_appear.colmap_id}' + '.mp4', mode='I', fps=30,codec='libx264',quality=10.0)#
                #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(view_appear.image_height), int(view_appear.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
        rendering = render(view_appear, gaussians, pipe=pipeline, bg_color=background,kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)["render"]
        for idx, view in enumerate(tqdm(generated_views, desc="Rendering progress")):
            view.camera_center=view_appear.camera_center
            gt_image = view.original_image.unsqueeze(0).cuda()
            gt_image_features=gaussians.app_encoder(normalize_vgg(gt_image))
            gt_image_features=gt_image_features.relu3_1
            rendered_feature = render(view, gaussians, pipe=pipeline, bg_color=background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)["render"] 
            tranfered_features = gaussians.style_transfer(
                rendered_feature.unsqueeze(0), #.detach(), # point cloud features [N, C]
                gt_image_features, 
            )
            rendering = gaussians.decoder(tranfered_features)
            rendering=rendering.squeeze()
            render_video_out.append_data(rendering.mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
            img_np=save_image(rendering, os.path.join(render_path, f"{name}_{appear_idx}_"+'{0:05d}'.format(idx) + ".png"))
            
            render_video_out.append_data(img_np)

        render_video_out.close()
       
 

def render_lego(model_path, name, iteration, views,view0, gaussians, pipeline, background):
    
    render_path = os.path.join(model_path, name,"ours_{}".format(iteration), f"renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    rendering = render(view0, gaussians, pipe=pipeline, bg_color=background,store_cache=True)["render"]
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def test_rendering_speed( views, gaussians, pipeline,background,use_cache=False):
    views=copy.deepcopy(views)
    length=100
    # view=views[0]
    for idx in range(length):
        view=views[idx] 
        view.original_image=torch.nn.functional.interpolate(view.original_image.unsqueeze(0),size=(800,800)).squeeze()
        view.image_height,view.image_width=800,800
    if not use_cache:
        rendering = render(views[0], gaussians, pipeline, background)["render"]
        start_time=time.time()
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx]
            rendering = render(view, gaussians, pipeline, background)["render"]
        end_time=time.time()
        
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    else:
        for i in range(100):
            views[i+1].image_height,views[i+1].image_width=view.image_height,view.image_width
        rendering = render(views[0], gaussians, pipeline, background,store_cache=True)["render"]
        start_time=time.time()
        rendering = render(view, gaussians, pipeline, background,store_cache=True)["render"]
        #for idx, view in enumerate(tqdm(views[1:], desc="Rendering progress")):
        for idx in tqdm(range(length), desc="Rendering progress"):
            view=views[idx+1]
            rendering = render(view, gaussians, pipeline, background,use_cache=True)["render"]       
        end_time=time.time()
        avg_rendering_speed=(end_time-start_time)/length
        print(f"rendering speed using cache:{avg_rendering_speed}s/image")
        return avg_rendering_speed
    
def render_intrinsic(model_path, name, iteration, views, gaussians, pipeline, background,):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_intrinsic")
    makedirs(render_path, exist_ok=True)
    gaussians.colornet_inter_weight=0.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]       
    
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    gaussians.colornet_inter_weight=1.0
    
def render_set(model_path, name, iteration, views, gaussians, pipeline, background,\
    render_multi_view=False,render_s2d_inter=False):
    '''
    '''
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    if gaussians.use_features_mask:
        mask_path=os.path.join(model_path, name, "ours_{}".format(iteration), "masks")
        makedirs(mask_path, exist_ok=True)

    if render_multi_view:
        multi_view_path=os.path.join(model_path, name, "ours_{}".format(iteration), "multi_view")
    if render_s2d_inter:
        s2d_inter_path=os.path.join(model_path, name, "ours_{}".format(iteration), "intrinsic_dynamic_interpolate")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    origin_views=copy.deepcopy(views)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]       
        gt = view.original_image[0:3, :, :]

        if gaussians.use_features_mask:
            tmask=gaussians.features_mask.repeat(1,3,1,1)
            torchvision.utils.save_image(tmask, os.path.join(mask_path, '{0:05d}'.format(idx) + ".png"))
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    if render_multi_view:
        #origin_views=copy.deepcopy(views)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            sub_multi_view_path=os.path.join(multi_view_path,f"{idx}")
            makedirs(sub_multi_view_path, exist_ok=True)
            for o_idx,o_view in enumerate(tqdm(origin_views, desc="Rendering progress")):
                rendering = render(view, gaussians, pipeline, background,\
                         other_viewpoint_camera=o_view)["render"]
                torchvision.utils.save_image(rendering, os.path.join(sub_multi_view_path, f"{idx}_{o_idx}" + ".png"))
    if render_s2d_inter and gaussians.color_net_type in ["naive"]:
        
        views=origin_views
        inter_weights=[i*0.1 for i in range(0,21)]
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            sub_s2d_inter_path=os.path.join(s2d_inter_path,f"{idx}")
            makedirs(sub_s2d_inter_path, exist_ok=True)
            for inter_weight in  inter_weights:
                gaussians.colornet_inter_weight=inter_weight
                rendering = render(view, gaussians, pipeline, background)["render"]
                torchvision.utils.save_image(rendering, os.path.join(sub_s2d_inter_path, f"{idx}_{inter_weight:.2f}" + ".png"))
        gaussians.colornet_inter_weight=1.0
        
    return 0
from scene.VGG import VGGEncoder, normalize_vgg
def render_sets(args, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt):
    with torch.no_grad():
        vgg_encoder= VGGEncoder().cuda()
        gaussians = GaussianModel(dataset.sh_degree)
        (model_params, first_iter) = torch.load(args.ckpt_path)
        gaussians.restore(model_params, opt, vgg_encoder, args)
        #iteration=1
        # dataset.resolution=args.resolution_
        scene = Scene(dataset, gaussians, resolution_scales=[1], shuffle=False, pretrained=True)  #

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        if args.render_multiview_vedio:
            render_multiview_vedio(dataset.model_path,"train", scene.getTrainCameras(),scene.getTestCameras(), gaussians, pipeline, background, args, dataset)
        if args.render_interpolate:
            #appearance tuning
            render_interpolate(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


        # gaussians.set_eval(False)
        
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    # model = ModelParams(parser, sentinel=True)
    # pipeline = PipelineParams(parser)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resolution_", default=2, type=int)
    parser.add_argument("--render_interpolate", action="store_true",default=False)
    
    parser.add_argument("--scene_name", type=str, default="brandenburg")
    
    parser.add_argument("--render_multiview_vedio", action="store_true",default=False)
    parser.add_argument('--masktype', type=str, default="maskrcnn", #maskrcnn context
                        help='mode seeking')
    parser.add_argument('--mask', action='store_true', default=True)
    parser.add_argument('--appearance', action='store_true', default=False)
    parser.add_argument("--model_path_args", type=str, default="output/test1")
    parser.add_argument('--encode_a_random', action='store_true', default=True)
    # dataset.kernel_size=config.kernel_size
    parser.add_argument("--kernel_size_", default=0.01, type=int)
    parser.add_argument("--ckpt_path", type=str, default="brandenburg")
    parser.add_argument('--eval_', action='store_true', default=True)
    
    # parser.add_argument("--model_path", type=str, default="output/test1")
    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)
    lp_=lp.extract(args)
    lp_.eval=args.eval_
    safe_state(args.quiet)
    lp_.kernel_size=args.kernel_size_
    lp_.resolution=2
    render_sets(args, lp_, args.iteration, pp.extract(args), args.skip_train, args.skip_test, op.extract(args))
