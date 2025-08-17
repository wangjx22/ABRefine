set -xe


PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../../

source activate /nfs_beijing/kubeflow-user/jinxian_2024
# commit_id=$(git rev-parse HEAD)
commit_id=None
# export PATH=/opt/conda/envs/gppi/bin:$PATH
export PATH=/nfs_beijing/kubeflow-user/jinxian_2024/bin/:$PATH

device_id=0
# set_random_seed(yaml_args['config_runtime'].seed)
# 输出的pdb综合数据csv存储路径
# refine pdb的保存路径
refine_pdb_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/pdb_file_trial5/'

input_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv'
ckpt_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/run_train_sin_k8s_trial3_2024.11.18-12.13.00_Step_58000_improve_rat_0.08204769827546084_imporve_num_138.ckpt'
output_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/'
yaml_path=${PROJECT_DIR}/yamls/refine_train_yamls/run_train_sin_k8s_trial5.yaml

# mkdir -p $output_path

CUDA_VISIBLE_DEVICES=${device_id} python ${PROJECT_DIR}/run_inference_sin.py \
  --input_path $input_path \
  --ckpt_path $ckpt_path \
  --output_path $output_path \
  --yaml-path $yaml_path \
  --refine_pdb_path $refine_pdb_path
