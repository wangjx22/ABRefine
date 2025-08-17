set -xe

source activate equiscore

PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../../

device_id=0

input_path=$1
ckpt_path=$2
output_dir=$3
output_path=${output_dir}/inference.txt
yaml_path=$4

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=${device_id} python ${PROJECT_DIR}/run_inference.py \
  --input_path=${input_path} \
  --ckpt_path=${ckpt_path} \
  --output_path=${output_path} \
  --yaml-path $yaml_path
