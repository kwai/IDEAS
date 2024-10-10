export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_MIN_NCHANNELS=16
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5

model=''

wandb disabled
deepspeed_config_path=''


nnodes=1

model_max_length=4096
per_device_train_batch_size=16
gradient_accumulation_steps=2

set -x

learning_rate=2e-5
rdzv=

echo ${per_device_train_batch_size}

output_dir="/mmu_nlp_ssd/wujiayu03/parrot_v2/output/skill_llama2_${model_max_length}_scaling_${data//\//-}"
output_dir="${output_dir::-5}_num1w"

nohup deepspeed --num_nodes 1 --num_gpus 8 --master_addr ${rdzv} --master_port=12346 train_llama_vicuna_syc.py \
    --deepspeed ${deepspeed_config_path} \
    --model_name_or_path ${model} \
    --data_path ${data} \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate ${learning_rate} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --model_max_length ${model_max_length} \
    --lazy_preprocess True \
    --group_by_length False \
    --dataloader_num_workers 16 \