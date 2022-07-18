python scripts/style_transfer.py \
--ckpt_path "/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
--input_filepaths \
"/import/c4dm-datasets/deepafx_st/vctk_24000/p314.wav" \
--style_filepaths \
"/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/val/broadcast/062_broadcast_cleanraw_val.wav" \
--save \
--modify_input \
--output_dir style_transfer_modify \

#"examples/obama.wav" \
#"examples/60min_presenter.wav" \

#python scripts/style_transfer.py \
#--ckpt_path "/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
#--input_filepaths \
#"/import/c4dm-datasets/deepafx_st/vctk_24000/p314.wav" \
#--style_filepaths \
#"/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/val/telephone/066_telephone_cleanraw_val.wav" \
#"/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/val/bright/061_bright_cleanraw_val.wav" \
#"/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/val/warm/067_warm_cleanraw_val.wav" \
#"/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/val/broadcast/060_broadcast_cleanraw_val.wav" \
#"/import/c4dm-datasets/deepafx_st/daps_24000_styles_100/val/neutral/069_neutral_cleanraw_val.wav" \
#--save \