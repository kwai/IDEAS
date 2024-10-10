export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_MIN_NCHANNELS=32
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

wandb disabled
export WANDB_MODE=offline


set -x
model='13b'
data=''
candidate=''

rdzv=
nnodes=8
num_gpus=8
model_max_length=4096

deepspeed_config_path=''
deepspeed_hostfile_path=''
init_model=''
global_batch_size=1024

if [[ $model == '13b' ]]; then
    learning_rate=2e-5
    per_device_train_batch_size=16
    gradient_accumulation_steps=$(( $global_batch_size / $per_device_train_batch_size / $nnodes / 8))
elif [[ $model == '30b' ]]; then
    learning_rate=1e-6
    per_device_train_batch_size=12
    gradient_accumulation_steps=$(( $global_batch_size / $per_device_train_batch_size / $nnodes / 8))
elif [[ $model == '7b' ]]; then
    learning_rate=2e-5
    per_device_train_batch_size=1
    gradient_accumulation_steps=$(( $global_batch_size / $per_device_train_batch_size / $nnodes / 8))
fi

output_dir="/skill_output_llama2_${model}_hf_${data//\//-}"
output_dir="${output_dir::-5}"

nohup deepspeed --num_nodes ${nnodes} --num_gpus ${num_gpus} --hostfile ${deepspeed_hostfile_path} --master_addr ${rdzv} --master_port=12344 train_user_simulator.py \
    --deepspeed ${deepspeed_config_path} \
    --model_name_or_path ${init_model} \
    --data_path ${data} \
    --candidate_path ${candidate} \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 100 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --model_max_length ${model_max_length} \
    --lazy_preprocess True \
    --group_by_length False \
    --dataloader_num_workers 128 \
