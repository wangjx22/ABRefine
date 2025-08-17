set -xe

source activate /nfs_beijing/kubeflow-user/jinxian_2024
# commit_id=$(git rev-parse HEAD)
commit_id=None
# export PATH=/opt/conda/envs/gppi/bin:$PATH
export PATH=/nfs_beijing/kubeflow-user/jinxian_2024/bin/:$PATH

wandb_key=b00f1fee3fce213cd72cfeb42122329d758bf176
wandb_name=NB_EquiformerV2_train
wandb login --relogin $wandb_key

PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../../
experiment_name=run_train_sin_k8s_trial4
yaml_path=${PROJECT_DIR}/yamls/refine_train_yamls/run_train_sin_k8s_trial4.yaml
device_ids=0         # not used for k8s running.
gpus=1
epochs=100       # not used, controlled by max_epochs
max_epochs=100
log_interval=200
eval_interval=2000
test_eval_interval=2000
save_interval=2000
start_decay_after_n_steps=50000
decay_every_n_steps=50000
warmup_no_steps=20000
base_lr=0
gradient_accumulation_steps=2
override_some_deepspeed_config="gradient_accumulation_steps"
max_lr=0.0001
output_dir=/nfs_beijing_ai/jinxian/rama-scoring1.3.0/trial4/
# mkdir -p ${output_dir}
#launch_mode=$1
launch_mode=k8s       # 设置成local应该就是本地跑，设置成k8s就是在k8s上分布跑

if [[ x$launch_mode == xlocal ]]; then
    CUDA_VISIBLE_DEVICES=${device_ids} deepspeed --include="localhost:${device_ids}" --master_port=12342 ${PROJECT_DIR}/run_train_sin.py \
        --deepspeed-init-dist \
        --deepspeed_config ${PROJECT_DIR}/deepspeed_config.json \
        --yaml-path $yaml_path \
        --epochs $epochs \
        --log_interval $log_interval \
        --eval_interval $eval_interval \
        --save_interval $save_interval \
        --strict_eval \
        --save $output_dir\
        --commit-id $commit_id \
        --wandb \
        --wandb-key $wandb_key \
        --wandb-name $wandb_name \
        --experiment-name $experiment_name \
        --max_epochs $max_epochs \
        --start_decay_after_n_steps $start_decay_after_n_steps \
        --decay_every_n_steps $decay_every_n_steps \
        --test_eval_interval $test_eval_interval \
        --warmup_no_steps $warmup_no_steps \
        --base_lr $base_lr \
        --max_lr $max_lr \
        --override_some_deepspeed_config $override_some_deepspeed_config \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --version 2 #> ./logs/run_train_debug.logs &
else
    gpu_count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
    echo "detect visiable gpus: $gpu_count"
    torchrun --nproc_per_node=${gpu_count} ${PROJECT_DIR}/run_train_sin.py \
        --deepspeed-init-dist \
        --deepspeed_config ${PROJECT_DIR}/deepspeed_config.json \
        --yaml-path $yaml_path \
        --epochs $epochs \
        --log_interval $log_interval \
        --eval_interval $eval_interval \
        --save_interval $save_interval \
        --strict_eval \
        --save $output_dir\
        --commit-id $commit_id \
        --wandb \
        --wandb-key $wandb_key \
        --wandb-name $wandb_name \
        --experiment-name $experiment_name \
        --max_epochs $max_epochs \
        --start_decay_after_n_steps $start_decay_after_n_steps \
        --decay_every_n_steps $decay_every_n_steps \
        --test_eval_interval $test_eval_interval \
        --warmup_no_steps $warmup_no_steps \
        --base_lr $base_lr \
        --max_lr $max_lr \
        --override_some_deepspeed_config $override_some_deepspeed_config \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --version 2 #> ./logs/run_train_debug.logs &
fi

# =====> END <======
