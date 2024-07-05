cd /root/young/code/mip-splatting/
for ks in {0.5,1,0.01,0.05,10}
do
python train-wildfeature.py -s /root/young/code/unconstrained-gs/data/brandenburg_gate -m  /root/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6210 --kernel_size $ks --model_path_args  output/ks_$ks 
done