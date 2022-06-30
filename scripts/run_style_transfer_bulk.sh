python scripts/style_transfer_bulk.py \
--ckpt_path "/import/c4dm-datasets/deepafx_st/logs/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
--input_filepaths \
"/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/telephone/810_telephone_cleanraw_val.wav" \
"/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/bright/801_bright_cleanraw_val.wav" \
"/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/warm/807_warm_cleanraw_val.wav" \
"/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/radio/803_radio_cleanraw_val.wav" \
"/import/c4dm-datasets/deepafx_st/daps_24000_styles/val/podcast/804_podcast_cleanraw_val.wav" \
--save \