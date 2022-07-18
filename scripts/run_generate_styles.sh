root_dir="/path/to/data" # path to audio datasets

python scripts/generate_styles.py \
"$root_dir/daps_24000/cleanraw" \
--output_dir "$root_dir/daps_24000_styles_100" \
--length_samples 131072 \
--num 100 \

python scripts/generate_styles.py \
"$root_dir/musdb18_44100" \
--output_dir "$root_dir/musdb18_44100_styles_100" \
--length_samples 262144 \
--num 100 \
