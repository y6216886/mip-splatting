# cd /root/young/code/mip-splatting/
# CUDA_VISIBLE_DEVICES=1 python train-wildfeature.py -s /root/young/code/unconstrained-gs/data/brandenburg_gate -m  /root/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6210 --kernel_size 0.1 --model_path_args  output/test --appearance --mask --masktype resnet18 --encode_a_random 

# export CUDA_HOME=/usr/local/cuda-11.4/

# export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}


TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6" pip install submodules/feature-gaussian-rasterization
TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6" pip install submodules/simple-knn/