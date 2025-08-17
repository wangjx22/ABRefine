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
refine_pdb_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/trial59/'

input_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_torchMD_split_seq/AF3_gen_pdb_files.csv'
ckpt_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/run_train_sin_k8s_trial59_yaml_run_train_sin_k8s_trial59_2025.05.14-15.01.03_Step_108000_validation_data_improve_rmsd_0.02486417132616043_imporve_num_466.0.ckpt'
output_path='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/inference/'
yaml_path=${PROJECT_DIR}/yamls/refine_train_yamls/run_train_sin_k8s_trial59.yaml

# mkdir -p $output_path

CUDA_VISIBLE_DEVICES=${device_id} python ${PROJECT_DIR}/run_inference_sin.py \
  --input_path $input_path \
  --ckpt_path $ckpt_path \
  --output_path $output_path \
  --yaml-path $yaml_path \
  --refine_pdb_path $refine_pdb_path
