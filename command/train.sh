cd /root/young/code/mip-splatting/
CUDA_VISIBLE_DEVICES=1 python train-wildfeature.py -s /root/young/code/unconstrained-gs/data/brandenburg_gate -m  /root/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6210 --kernel_size 0.1 --model_path_args  output/test --appearance --mask --masktype resnet18 --encode_a_random 
