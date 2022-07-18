gpu_id=0
num_examples=1000 # number of evaluation examples per dataset
checkpoint_dir="./checkpoints"
root_dir="/path/to/data" # path to audio datasets
output_dir="/path/to/data/eval" # path to store audio outputs

# ----------------------- LibriTTS ----------------------- 
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/libritts/" \
--root_dir "$root_dir" \
--gpu \
--dataset libritts \
--dataset_dir "LibriTTS/train_clean_360_24000c" \
--spsa_version 2 \
--tcn1_version 1 \
--autodiff_version 1 \
--tcn2_version 1 \
--subset "test" \
--output "$output_dir" \
--examples "$num_examples" \
--save \

# ----------------------- DAPS ----------------------- 
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/libritts/" \
--root_dir "$root_dir" \
--gpu \
--dataset daps \
--dataset_dir "daps_24000/cleanraw" \
--spsa_version 2 \
--tcn1_version 1 \
--autodiff_version 1 \
--tcn2_version 1 \
--subset "train" \
--output "$output_dir" \
--examples "$num_examples" \
--save \

# ----------------------- VCTK ----------------------- 
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/libritts/" \
--root_dir "$root_dir" \
--gpu \
--dataset vctk \
--dataset_dir "vctk_24000" \
--spsa_version 2 \
--tcn1_version 1 \
--autodiff_version 1 \
--tcn2_version 1 \
--subset "train" \
--examples "$num_examples" \
--output "$output_dir" \

# ----------------------- Jamendo @ 24kHz (test) -----------------------
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/jamendo/" \
--root_dir "$root_dir" \
--gpu \
--dataset jamendo \
--dataset_dir "mtg-jamendo_24000/" \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset test \
--save \
--ext flac \
--examples "$num_examples" \
--output "$output_dir" \

# ----------------------- Jamendo @ 24kHz (test) -----------------------
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/jamendo/" \
--root_dir "$root_dir" \
--gpu \
--dataset jamendo_44100 \
--dataset_dir "mtg-jamendo_44100/" \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset test \
--length 262144 \
--save \
--ext wav \
--examples "$num_examples" \
--output "$output_dir" \

# -----------------------  MUSDB18 @ 24kHz (train) -----------------------
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/jamendo/" \
--root_dir "$root_dir" \
--gpu \
--dataset musdb18_24000 \
--dataset_dir "musdb18_24000/" \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset train \
--length 131072 \
--save \
--ext wav \
--examples "$num_examples" \
--output "$output_dir" \

# -----------------------  MUSDB18 @ 44.1kHz (train) -----------------------
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/eval_style.py \
"$checkpoint_dir/style/jamendo/" \
--root_dir "$root_dir" \
--gpu \
--dataset musdb18_44100 \
--dataset_dir "musdb18_44100/" \
--spsa_version 0 \
--tcn1_version 0 \
--autodiff_version 0 \
--tcn2_version 0 \
--subset train \
--length 262144 \
--save \
--ext wav \
--examples "$num_examples" \
--output "$output_dir" \

# ----------------------- Style case study (SPSA) -----------------------
## Style case study on DAPS
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/style_case_study.py \
--ckpt_paths \
"$checkpoint_dir/style/libritts/spsa/lightning_logs/version_2/checkpoints/epoch=367-step=1226911-val-libritts-spsa.ckpt" \
"$checkpoint_dir/style/libritts/autodiff/lightning_logs/version_1/checkpoints/epoch=367-step=1226911-val-libritts-autodiff.ckpt" \
"$checkpoint_dir/style/libritts/proxy0/lightning_logs/version_0/checkpoints/epoch=327-step=1093551-val-libritts-proxy0.ckpt" \
--style_audio "$root_dir/daps_24000_styles_1000_diverse/train" \
--output_dir "$root_dir/style_case_study" \
--gpu \
--save \
--plot \

## Style case study on MUSDB18 @ 44.1 kHz
CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/style_case_study.py \
--ckpt_paths \
"$checkpoint_dir/style/jamendo/autodiff/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-autodiff.ckpt" \
"$checkpoint_dir/style/jamendo/spsa/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-spsa.ckpt" \
"$checkpoint_dir/style/jamendo/proxy0/lightning_logs/version_0/checkpoints/epoch=362-step=1210241-val-jamendo-proxy0.ckpt" \
--style_audio "$root_dir/musdb18_44100_styles_100/train" \
--output_dir "$root_dir/style_case_study_musdb18" \
--sample_rate 44100 \
--gpu \
--save \
--plot \
