cd /root/young/code/mip-splatting/
python train-wildfeature.py -s /root/young/code/unconstrained-gs/data/brandenburg_gate -m  /root/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6210 --kernel_size 0.1 --model_path_args  output/app-random-app-maskrcnn-tune --appearance --mask --masktype maskrcnn --encode_a_random 
