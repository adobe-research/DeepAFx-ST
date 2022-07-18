#!/bin/bash

root_data_dir=/path/to/data
multi_gpu=0 # set to 1 to launch on sequential GPUs
gpu_id=0 # starting GPU id

for processor in  peq comp channel
do

   echo "Training $processor proxy on GPU $gpu_id"

    # Single Parametric EQ
    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/train_proxy.py \
    --gpus 1 \
    --input_dir "$root_data_dir/LibriTTS/train_clean_360_24000c" \
    --sample_rate 24000 \
    --train_length 65536 \
    --train_examples_per_epoch 20000 \
    --val_length 65536 \
    --val_examples_per_epoch 200 \
    --buffer_size_gb 1.0 \
    --buffer_reload_rate 2000 \
    --processor $processor \
    --causal \
    --output_gain \
    --nblocks 4 \
    --dilation_growth 8 \
    --channel_width 64 \
    --kernel_size 13 \
    --lr 3e-4 \
    --lr_patience 10 \
    --num_workers 8 \
    --batch_size 16 \
    --gradient_clip_val 10.0 \
    --max_epochs 400 \
    --accelerator ddp \
    --default_root_dir "$root_data_dir/logs/proxies/libritts/$processor"

    # set the GPU ID
   if [ $multi_gpu -eq 1 ]; then
      ((gpu_id=gpu_id+1))
   fi

done