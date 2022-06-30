python scripts/style_interpolation.py \
--ckpt_path "/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
--input_audio "/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/telephone/799_telephone_cleanraw_val.wav" \
--input_length 10 \
--style_a "/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/warm/822_warm_cleanraw_val.wav" \
--style_a_name "warm" \
--style_b "/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/radio/806_radio_cleanraw_val.wav" \
--style_b_name "radio" \
--save \

# "/import/c4dm-datasets/deepafx_st/vctk_24000/p314.wav"