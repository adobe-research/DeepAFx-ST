checkpoint_dir="./checkpoints"
root_dir="/path/to/data" # path to audio datasets
output_dir="/path/to/data/eval" # path to store audio utputs

CUDA_VISIBLE_DEVICES=0 python scripts/eval_probes.py \
--ckpt_dir "$checkpoint_dir/probes/speech" \
--eval_dataset "$root_dir/daps_24000_styles_100/" \
--subset test \
--audio_type speech \
--output_dir probes \
--gpu \

CUDA_VISIBLE_DEVICES=0 python scripts/eval_probes.py \
--ckpt_dir "$checkpoint_dir/probes/music" \
--eval_dataset "$root_dir/musdb18_44100_styles_100/" \
--audio_type music \
--subset test \
--output_dir probes \
--gpu \