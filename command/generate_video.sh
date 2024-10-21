# sacre coeur
# CUDA_VISIBLE_DEVICES=0 python ./render_videos.py  --model_path outputs/sacre/full --skip_train --skip_test --render_multiview_vedio
cd /root/young/code/mip-splatting
# CUDA_VISIBLE_DEVICES=2  python train-wildfeature.py -s /root/young/code/unconstrained-gs/data/brandenburg_gate -m /root/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6334 --model_path_args  /root/young/code/mip-splatting/output/test1 --appearance --mask --masktype maskrcnn --encode_a_random  --iterations_ 30000  #--start_checkpoint /root/young/code/mip-splatting/output/test1_k72sbzkb/chkpnt2000.pth


python /root/young/code/mip-splatting/render_videos.py --model_path /root/young/code/mip-splatting/output/brandenburggate --skip_train --skip_test --render_multiview_vedio -s /root/young/code/unconstrained-gs/data/brandenburg_gate --ckpt_path /root/young/code/mip-splatting/output/test1_8sx98wqy/chkpnt30000.pth