cd /root/young/code/mip-splatting/

dataset_name=brandenburg_gate #should work in path
source_path=/root/young/code/unconstrained-gs/data/$dataset_name


# for i in {1,2,3,4,5,6,7,8,9,10}
# do
cd /root/young/code/mip-splatting/
# ####
meta_name="pretrained_masknet1"
# name=default



model_path=output/$dataset_name/wild/$meta_name
echo $model_path
# "./output", os.path.basename(args.source_path), "wild", args.exp_name
command="CUDA_VISIBLE_DEVICES=0  python train-wildfeature.py -s /root/young/code/unconstrained-gs/data/brandenburg_gate -m /root/young/code/mip-splatting/output/brandenburggate --eval --load_allres --sample_more_highres --white_background --port 6322 --model_path_args_  $model_path --appearance --mask --masktype context --encode_a_random"



codepath=$model_path/codes
target_directory=$codepath

# Create a timestamp for the filename
timestamp=$(date +"%Y%m%d%H%M%S")

# Filename
mkdir -p $target_directory
filename="${timestamp}_command.txt"
# log_filepath="$target_directory/$filename"
# Save the command to the target directory
cp -r utils/ "$codepath/utils/"
cp -r scene/ "$codepath/scene/"
# cp -r data/ "$codepath/data/"
cp -r gaussian_renderer/ "$codepath/gaussian_renderer/"
cp -r lpipsPyTorch/ "$codepath/lpipsPyTorch/"
cp -r arguments/ "$codepath/arguments/"
cp eval_metric.py "$codepath/eval_metric.py"
cp metrics.py "$codepath/metrics.py"
cp train-wildfeature.py "$codepath/train-wildfeature.py"

cd $codepath
# echo "$command" > $log_filepath
# Execute the command
eval "$command"
# done
# done