CUDA_VISIBLE_DEVICES=0 python scripts/style_case_study.py \
--ckpt_paths \
"/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
"/import/c4dm-datasets/deepafx_st/logs/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt" \
"/import/c4dm-datasets/deepafx_st/logs/style/libritts/proxy0/lightning_logs/version_0/checkpoints/epoch=327-step=1093551-val-libritts-proxy0.ckpt" \
--style_audio "/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/train" \
--output_dir "/import/c4dm-datasets/deepafx_st/style_case_study_daps" \
--num_examples 1000 \
--gpu \
--save \
--plot \

#CUDA_VISIBLE_DEVICES=1 python scripts/style_case_study.py \
#--ckpt_paths \
#"/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt" \
#"/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/spsa/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-spsa.ckpt" \
#"/import/c4dm-datasets/deepafx_st/logs_jamendo/style/jamendo/proxy0/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-proxy0.ckpt" \
#--style_audio "/import/c4dm-datasets/deepafx_st/musdb18_44100_styles_100/train" \
#--output_dir "/import/c4dm-datasets/deepafx_st/style_case_study_musdb18" \
#--sample_rate 44100 \
#--gpu \
#--save \
#--plot \