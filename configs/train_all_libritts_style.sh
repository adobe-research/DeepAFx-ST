#!/bin/bash

root_data_dir=/path/to/data
multi_gpu=0 # set to 1 to launch on sequential GPUs
gpu_id=0 # starting GPU id 
# by default start on GPU #1 (id=0)
checkpoint_dir="./checkpoints"

for processor_model in  tcn1 tcn2 spsa proxy0 proxy2 autodiff
do

   if [ "$processor_model" = "tcn1" ]; then
      lr=1e-4
   elif [ "$processor_model" = "tcn2" ]; then
      lr=1e-4
   elif [ "$processor_model" = "spsa" ]; then
      lr=1e-5 # lower learning rate
   elif [ "$processor_model" = "proxy0" ]; then
      lr=1e-4
   elif [ "$processor_model" = "proxy1" ]; then
      lr=1e-4
   elif [ "$processor_model" = "proxy2" ]; then
      lr=1e-4
   elif [ "$processor_model" = "autodiff" ]; then
      lr=1e-4
   else
      lr=1e-4
   fi

   echo "Training $processor_model on GPU $gpu_id with learning rate = $lr"

   CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/train_style.py \
   --processor_model $processor_model \
   --gpus 1 \
   --audio_dir "$root_data_dir/LibriTTS" \
   --input_dirs "train_clean_360_24000c" \
   --style_transfer \
   --buffer_size_gb 1.0 \
   --buffer_reload_rate 2000 \
   --train_frac 0.9 \
   --freq_corrupt  \
   --drc_corrupt \
   --sample_rate 24000 \
   --train_length 131072 \
   --train_examples_per_epoch 20000 \
   --val_length 131072 \
   --val_examples_per_epoch 200 \
   --random_scale_input \
   --encoder_model efficient_net \
   --encoder_embed_dim 1024 \
   --encoder_width_mult 1 \
   --recon_losses mrstft l1 \
   --recon_loss_weight 1.0 100.0 \
   --tcn_causal \
   --tcn_nblocks 4 \
   --tcn_dilation_growth 8 \
   --tcn_channel_width 64 \
   --tcn_kernel_size 13 \
   --spsa_epsilon 0.0005 \
   --spsa_verbose \
   --spsa_parallel \
   --proxy_ckpts \
    "$checkpoint_dir/proxies/libritts/peq/lightning_logs/version_1/checkpoints/epoch=111-step=139999-val-libritts-peq.ckpt" \
    "$checkpoint_dir/proxies/libritts/comp/lightning_logs/version_1/checkpoints/epoch=255-step=319999-val-libritts-comp.ckpt" \
   --freeze_proxies \
   --lr "$lr" \
   --num_workers 8 \
   --batch_size 6 \
   --gradient_clip_val 4.0 \
   --max_epochs 400 \
   --accelerator ddp \
   --default_root_dir "$root_data_dir/logs/style/libritts/$processor_model" \

   # set the GPU ID
   if [ $multi_gpu -eq 1 ]; then
      ((gpu_id=gpu_id+1))
   fi
done