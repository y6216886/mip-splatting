# sacre coeur
# CUDA_VISIBLE_DEVICES=0 python ./render_videos.py  --model_path outputs/sacre/full --skip_train --skip_test --render_multiview_vedio
cd /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting
# CUDA_VISIBLE_DEVICES=2  python train-wildfeature.py -s /U_20240109_SZR_SMIL/yyf/young/code/unconstrained-gs/data/brandenburg_gate -m /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6334 --model_path_args  /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/test1 --appearance --mask --masktype maskrcnn --encode_a_random  --iterations_ 30000  #--start_checkpoint /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/test1_k72sbzkb/chkpnt2000.pth
export WANDB_API_KEY='23509f25f38fd1de301b370ab89f8ea19ac2e2c7'

python render_videos.py --model_path /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/brandenburggate --skip_train --skip_test --render_multiview_vedio -s /U_20240109_SZR_SMIL/yyf/young/code/unconstrained-gs/data/brandenburg_gate --ckpt_path /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/test1_eternal-sweep-1/chkpnt30000.pth



python render_videos.py --model_path /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/brandenburggate --skip_train --skip_test --render_multiview_vedio -s /U_20240109_SZR_SMIL/yyf/young/code/vggt/datasets/nerf_llff_data/trex/images --ckpt_path /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/test1_eternal-sweep-1/chkpnt30000.pth

DJI_20200223_163642_330
DJI_20200223_163643_411

# python render_videos.py --model_path /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/brandenburggate --skip_train --skip_test --render_multiview_vedio -s /U_20240109_SZR_SMIL/yyf/young/code/unconstrained-gs/data/brandenburg_gate --ckpt_path /U_20240109_SZR_SMIL/yyf/young/code/mip-splatting/output/brandenburg_gate/wild/mask_resnet18/codes/output/brandenburg_gate/wild/mask_resnet18_silvery-sweep-48/chkpnt30000.pth