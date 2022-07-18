#!/bin/bash

root_data_dir=/path/to/data
multi_gpu=0 # set to 1 to launch on sequential GPUs
gpu_id=0 # starting GPU id
checkpoint_dir="./checkpoints"

# random_mel openl3 deepafx_st_spsa deepafx_st_proxy0 deepafx_st_autodiff cdpam

probe_type=linear # always use linear probe

for audio_type in  speech
do
   if [ "$audio_type" = "speech" ]; then
      audio_dir="daps_24000_styles_100"
      deepafx_st_autodiff_ckpt="$checkpoint_dir/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt"
      deepafx_st_spsa_ckpt="$checkpoint_dir/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt"
      deepafx_st_proxy0_ckpt="$checkpoint_dir/style/libritts/proxy0/lightning_logs/version_0/checkpoints/epoch=327-step=1093551-val-libritts-proxy0.ckpt"
   elif [ "$audio_type" = "music" ]; then
      audio_dir="musdb18_44100_styles_100"
      deepafx_st_autodiff_ckpt="$checkpoint_dir/jamendo/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt"
      deepafx_st_spsa_ckpt="$checkpoint_dir/jamendo/style/jamendo/spsa/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-spsa.ckpt"
      deepafx_st_proxy0_ckpt="$checkpoint_dir/jamendo/style/jamendo/proxy0/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-proxy0.ckpt"
   fi

   for encoder_type in  random_mel openl3 deepafx_st_spsa deepafx_st_proxy0 deepafx_st_autodiff cdpam
   do

      if [ "$encoder_type" = "deepafx_st_autodiff" ]; then
         lr=1e-3
         encoder_sample_rate=24000
      elif [ "$encoder_type" = "deepafx_st_spsa" ]; then
         lr=1e-3
         encoder_sample_rate=24000
      elif [ "$encoder_type" = "deepafx_st_proxy0" ]; then
         lr=1e-3
         encoder_sample_rate=24000
      elif [ "$encoder_type" = "random_mel" ]; then
         lr=1e-3
         encoder_sample_rate=24000
      elif [ "$encoder_type" = "openl3" ]; then
         lr=1e-3
         encoder_sample_rate=48000
      elif [ "$encoder_type" = "cdpam" ]; then
         lr=1e-3
         encoder_sample_rate=22050
      else
         lr=1e-3
      fi

      echo "Training $audio_type $encoder_type encoder with $probe_type probe on GPU $gpu_id"

      CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/train_probe.py \
      --gpus 1 \
      --task "style" \
      --audio_dir "$root_data_dir/$audio_dir" \
      --sample_rate 24000 \
      --encoder_sample_rate "$encoder_sample_rate" \
      --encoder_type $encoder_type \
      --deepafx_st_autodiff_ckpt "$deepafx_st_autodiff_ckpt" \
      --deepafx_st_spsa_ckpt "$deepafx_st_spsa_ckpt" \
      --deepafx_st_proxy0_ckpt "$deepafx_st_proxy0_ckpt" \
      --cdpam_ckpt  "$checkpoint_dir/cdpam/scratchJNDdefault_best_model.pth" \
      --probe_type $probe_type \
      --lr "$lr" \
      --num_workers 4 \
      --batch_size 16 \
      --gradient_clip_val 200.0 \
      --max_epochs 400 \
      --accelerator ddp \
      --default_root_dir "$root_data_dir/probes/$audio_type/$encoder_type-$probe_type" \

      # set the GPU ID
      if [ $multi_gpu -eq 1 ]; then
         ((gpu_id=gpu_id+1))
      fi

   done
done
