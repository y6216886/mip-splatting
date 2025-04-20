# conda create -y -n mip-splatting python=3.9
# conda activate mip-splatting

# pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# conda install cudatoolkit-dev=11.3 -c conda-forge

# pip install -r requirements.txt

# pip install submodules/diff-gaussian-rasterization
# pip install submodules/simple-knn/
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/usr/local/cuda-11.4

export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}

conda create -n mip-splatting python=3.10 -y
conda activate mip-splatting
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system