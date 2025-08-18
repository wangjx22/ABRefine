set -xe


PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../../

source activate your env
# commit_id=$(git rev-parse HEAD)
commit_id=None
# export PATH=/opt/conda/envs/gppi/bin:$PATH
export PATH=./bin/:$PATH

device_id=0

refine_pdb_path='./dataset/datasets/inference/trial59/'

input_path='./dataset/datasets/AF3_gen_pdb_files.csv'
ckpt_path='./model/ckpt/run_train_sin_k8s_trial59_yaml_run_train_sin_k8s_trial59.ckpt'
output_path='./dataset/datasets/inference/'
yaml_path=${PROJECT_DIR}/yamls/refine_train_yamls/run_train_sin_k8s_trial59.yaml

# mkdir -p $output_path

CUDA_VISIBLE_DEVICES=${device_id} python ${PROJECT_DIR}/run_inference_sin.py \
  --input_path $input_path \
  --ckpt_path $ckpt_path \
  --output_path $output_path \
  --yaml-path $yaml_path \
  --refine_pdb_path $refine_pdb_path
