#!/usr/bin/env python3
# *************************************************************************
#  Copyright 2021 Adobe Systems Incorporated.
#
#  Please see the attached LICENSE file for more information.
#
# **************************************************************************/

import os
import sox
import wget
import glob
import torch
import shutil
import resampy
import hashlib
import itertools
import subprocess
import torchaudio
import numpy as np
import multiprocessing
import soundfile as sf

from tqdm import tqdm
from deepafx_st import utils
from argparse import ArgumentParser
from joblib import Parallel, delayed


def resample_file(spkr_file, sr):
    x, sr_orig = sf.read(spkr_file)
    if sr_orig != sr:
        x = resampy.resample(x, sr_orig, sr, axis=0)
    return x


def ffmpeg_resample(input_avfile, output_audiofile, sr, channels=None):
    cmd = ["ffmpeg", "-y", "-i", input_avfile,
        "-ar", str(sr), "-ac", str(1), output_audiofile,
        "-hide_banner", "-loglevel", "error", ]
    completed_process = subprocess.run(cmd)
    return completed_process


def resample_dir(input_dir, output_dir, target_sr, channels=1, num_cpus=33):

    # files = get_audio_file_list(input_dir)
    files = glob.glob(os.path.join(input_dir, "*.mp3"))

    os.makedirs(output_dir, exist_ok=True)

    file_pairs = []
    for file in files:
        new_file = os.path.join(output_dir, os.path.basename(file)[:-4] + ".wav")
        file_pairs.append((file, new_file, target_sr))

    def par_resample(item):
        orig, new, sr = item
        ffmpeg_resample(orig, new, sr, channels=1)
        return True

    # FFMPEG seems to have issue when multi-threaded
    # results = Parallel(n_jobs=num_cpus)(
    #     delayed(par_resample)(i) for i in tqdm(file_pairs)
    # )
    for item in tqdm(file_pairs):
        par_resample(item)



def resample_file_torchaudio(spkr_file, sr):
    x, sr_orig = torchaudio.load(spkr_file)
    x = x.numpy()
    if sr_orig != sr:
        x = resampy.resample(x, sr_orig, sr)
    x = torch.tensor(x)
    return x


def download_daps_dataset(output_dir):
    """Download and resample the DAPS dataset to a given sample rate."""

    archive_path = os.path.join(output_dir, "daps.tar.gz")

    cmd = f"wget -O {archive_path} https://zenodo.org/record/4660670/files/daps.tar.gz?download=1"
    os.system(cmd)

    # Untar
    print("Extracting tar...")
    cmd = f"tar -xvf {archive_path} -C {output_dir}"
    os.system(cmd)


def process_daps_dataset(output_dir):

    set_dirs = glob.glob(os.path.join(output_dir, "daps", "*"))

    for sr in [16000, 24000, 44100]:
        resampled_output_dir = os.path.join(output_dir, f"daps_{sr}")
        if not os.path.isdir(resampled_output_dir):
            os.makedirs(resampled_output_dir)

        for set_dir in set_dirs:
            print(set_dir)
            if "produced" in set_dir or "cleanraw" in set_dir:
                # get all files in speaker directory
                spkr_files = glob.glob(os.path.join(set_dir, "*.wav"))

                with multiprocessing.Pool(16) as pool:
                    audios = pool.starmap(
                        resample_file,
                        zip(spkr_files, itertools.repeat(sr)),
                    )

                for spkr_file, audio in tqdm(zip(spkr_files, audios)):
                    spkr_id = os.path.basename(spkr_file)

                    if not os.path.isdir(
                        os.path.join(
                            resampled_output_dir, f"{os.path.basename(set_dir)}"
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                resampled_output_dir, f"{os.path.basename(set_dir)}"
                            )
                        )

                    out_filepath = os.path.join(
                        resampled_output_dir,
                        f"{os.path.basename(set_dir)}",
                        f"{spkr_id}",
                    )
                    sf.write(out_filepath, audio, sr)

def download_vctk_dataset(output_dir):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    archive_path = os.path.join(output_dir, "vctk.zip")

    cmd = (
        f"wget -O {archive_path} https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
    )
    os.system(cmd)

    # Untar
    print("Extracting zip...")
    cmd = f"unzip {archive_path} -C {output_dir}"
    os.system(cmd)


def process_vctk_dataset(output_dir, num_workers=16):

    spkr_dirs = glob.glob(
        os.path.join(
            output_dir,
            "VCTK-Corpus-0.92",
            "wav48_silence_trimmed",
            "*",
        )
    )

    for sr in [16000, 24000, 44100]:

        resampled_output_dir = os.path.join(output_dir, f"vctk_{sr}")
        if not os.path.isdir(resampled_output_dir):
            os.makedirs(resampled_output_dir)

        for spkr_dir in tqdm(spkr_dirs, ncols=80):
            print(spkr_dir)
            # get all files in speaker directory
            spkr_files = glob.glob(os.path.join(spkr_dir, "*.flac"))
            spkr_id = os.path.basename(spkr_dir)

            if len(spkr_files) > 0:
                with multiprocessing.Pool(num_workers) as pool:
                    audios = pool.starmap(
                        resample_file,
                        zip(spkr_files, itertools.repeat(sr)),
                    )
                # combine all audio files into one long file
                x = np.concatenate(audios, axis=-1)

                out_filepath = os.path.join(resampled_output_dir, f"{spkr_id}.wav")
                sf.write(out_filepath, x, sr)
            else:
                print(f"{spkr_dir} contained no audio files.")


def download_libritts_dataset(output_dir, sr=24000):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    archive_path = os.path.join(output_dir, "train-clean-360.tar.gz")

    cmd = f"wget -O {archive_path} https://www.openslr.org/resources/60/train-clean-360.tar.gz"
    os.system(cmd)

    # Untar
    print("Extracting tar...")
    cmd = f"tar -xvf {archive_path} -C {output_dir}"
    os.system(cmd)


def process_libritts_dataset(output_dir):

    spkr_dirs = glob.glob(
        os.path.join(
            output_dir,
            "LibriTTS",
            "train-clean-360",
            "*",
        )
    )

    for sr in [16000, 24000]:

        resampled_output_dir = os.path.join(
            output_dir,
            "LibriTTS",
            f"train_clean_360_{sr}c",
        )
        if not os.path.isdir(resampled_output_dir):
            os.makedirs(resampled_output_dir)

        for spkr_dir in tqdm(spkr_dirs, ncols=80):
            # get all book directories
            spkr_id = os.path.basename(spkr_dir)
            book_dirs = glob.glob(os.path.join(spkr_dir, "*"))

            spkr_files = []
            for book_dir in book_dirs:
                # get all files in speaker directory
                spkr_files += glob.glob(os.path.join(book_dir, "*.wav"))
            print(
                f"Found {len(book_dirs)} books with {len(spkr_files)} files by {spkr_id}"
            )

            if len(spkr_files) > 0:
                with multiprocessing.Pool(16) as pool:
                    audios = pool.starmap(
                        resample_file,
                        zip(spkr_files, itertools.repeat(sr)),
                    )
                # combine all audio files into one long file
                x = np.concatenate(audios, axis=-1)
                # print(x.shape, (x.shape[0] / sr) / 60)

                out_filepath = os.path.join(resampled_output_dir, f"{spkr_id}.wav")
                sf.write(out_filepath, x, sr)
            else:
                print(f"{spkr_dir} contained no audio files.")


def download_jamendo_dataset(output_dir):

    hash_url = "https://essentia.upf.edu/datasets/mtg-jamendo/autotagging_moodtheme/audio/checksums_sha256.txt"
    cmd = f"""wget -O {os.path.join(output_dir, "checksums_sha256.txt")} {hash_url}"""
    os.system(cmd)

    with open(os.path.join(output_dir, "checksums_sha256.txt"), "r") as fp:
        hashes = fp.readlines()

    hash_dict = {}
    for sha256_hash in hashes:
        value = sha256_hash.split(" ")[0]
        fname = sha256_hash.split(" ")[1].strip("\n")
        hash_dict[fname] = value

    for n in range(100):
        base_url = (
            "https://essentia.upf.edu/datasets/mtg-jamendo/autotagging_moodtheme/audio/"
        )
        fname = f"autotagging_moodtheme_audio-{n:02}.tar"
        url = base_url + fname
        # check if file has been downloaded
        if os.path.isfile(os.path.join(output_dir, fname)):

            # comute hash for downloaded file
            sha256_hash = check_sha256(os.path.join(output_dir, fname))

            # check this against out dictionary
            if sha256_hash == hash_dict[fname]:
                print(f"Checksum PASSED. Skipping {fname}...")
                continue
            else:
                print("Checksum FAILED. Re-downloading...")

        cmd = f"wget -O {os.path.join(output_dir, fname)} {url}"
        os.system(cmd)

    for n in range(100):
        fname = f"autotagging_moodtheme_audio-{n:02}.tar"
        # Untar
        print(f"Extracting {fname}...")
        cmd = f"tar -xvf {os.path.join(output_dir, fname)} -C {output_dir}"
        os.system(cmd)

def process_jamendo_dataset(output_dir):

    num_cpus = multiprocessing.cpu_count()
    set_dirs = []
    for n in range(100):
        set_dirs.append(os.path.join(output_dir, str(n)))
    
    for sr in [24000]:
        resampled_output_dir = os.path.join(output_dir, f"mtg-jamendo_{sr}")
        if not os.path.isdir(resampled_output_dir):
            os.makedirs(resampled_output_dir)

        for set_dir in set_dirs:
            # get all files in speaker directory
            resample_dir(set_dir, resampled_output_dir, sr, channels=1, num_cpus=num_cpus)


def download_musdb_dataset(output_dir):
    # from https://zenodo.org/record/3338373.
    cmd = 'wget https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1 -O ' + os.path.join(output_dir, 'musdb18.zip')
    os.system(cmd)

    cmd = 'unzip  ' + os.path.join(output_dir, 'musdb18.zip') + ' -d ' + os.path.join(output_dir, 'musdb18')
    os.system(cmd)
    

def process_musdb_dataset(output_dir):
    
    def resample_file(item):
        orig, new, sr = item
        x, sr_orig = sf.read(orig)
        if sr_orig != sr:
            x = resampy.resample(x, sr_orig, sr, axis=0)
        
        sf.write(new, x, sr)
        return True

    
    mix_files = glob.glob(os.path.join(output_dir, 'musdb18', "**", "*.wav"), recursive=True)
    mix_files = [mix_file for mix_file in mix_files if "mix" in mix_file]

    items = []
    for sr in [24000, 44100]:
        resampled_output_dir = os.path.join(output_dir, f"musdb18_{sr}")
        if not os.path.isdir(resampled_output_dir):
            os.makedirs(resampled_output_dir)
        
        for mix_file in mix_files:
            song_id = os.path.basename(os.path.dirname(mix_file)).replace(" ", "")
            out_filepath = os.path.join(
                resampled_output_dir,
                f"{song_id}.wav",
            )
            items.append((mix_file, out_filepath, sr))
 
        
    num_cpus = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cpus)(
        delayed(resample_file)(i) for i in tqdm(items)
    )



if __name__ == "__main__":

    parser = ArgumentParser(description="Download all models and datasets.")
    parser.add_argument(
        "--checkpoint",
        help="Download pre-trained model checkpoints.",
        action="store_true",
    )
    parser.add_argument(
        "--datasets",
        help="Datasets to download.",
        nargs="+",
        default=[
            "daps",
            "vctk",
            "jamendo",
            "libritts",
            "musdb",
        ],
    )
    parser.add_argument(
        "-d",
        "--download",
        help="Download the dataset.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--process",
        help="Process the dataset assuming it is already downloaded.",
        action="store_true",
    )
    parser.add_argument(
        "--output",
        help="Root directory to download dataset.",
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        help="Number of parallel workers",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = "./"

    if not os.path.isdir(args.output):
        os.makedirs(args.output)

    for dataset in args.datasets:
        if dataset == "daps":
            
            if args.download:
                print("Downloading DAPS...")
                download_daps_dataset(args.output)
            if args.process:
                print(f"Processing DAPS dataset...")
                process_daps_dataset(args.output)
        elif dataset == "vctk":
            if args.download:
                print("Downloading VCTK...")
                download_vctk_dataset(args.output)
            if args.process:
                print(f"Processing VCTK dataset...")
                process_vctk_dataset(args.output)
        elif dataset == "libritts":
            
            if args.download:
                print("Downloading LibriTTS...")
                download_libritts_dataset(args.output)
            if args.process:
                print(f"Processing libriTTS dataset...")
                process_libritts_dataset(args.output)
        elif dataset == "jamendo":
            
            if args.download:
                print(f"Downloading Jamendo dataset...")
                download_jamendo_dataset(args.output)
            if args.process:
                print(f"Processing Jamendo dataset...")
                process_jamendo_dataset(args.output)
        elif dataset == "musdb":
            
            if args.download:
                print(f"Downloading MUSDB dataset...")
                download_musdb_dataset(args.output)
            if args.process:
                print(f"Processing MUSDB dataset...")
                process_musdb_dataset(args.output)
        else:
            print("\nInvalid dataset.\n")
            parser.print_help()
