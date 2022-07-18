import os
import sys
import csv
import glob
import torch
import random
from tqdm import tqdm
from typing import List, Any

from deepafx_st.data.audio import AudioFile
import deepafx_st.utils as utils
import deepafx_st.data.augmentations as augmentations


class AudioDataset(torch.utils.data.Dataset):
    """Audio dataset which returns an input and target file.

    Args:
        audio_dir (str): Path to the top level of the audio dataset.
        input_dir (List[str], optional): List of paths to the directories containing input audio files. Default: ["clean"]
        subset (str, optional): Dataset subset. One of ["train", "val", "test"]. Default: "train"
        length (int, optional): Number of samples to load for each example. Default: 65536
        train_frac (float, optional): Fraction of the files to use for training subset. Default: 0.8
        val_frac (float, optional): Fraction of the files to use for validation subset. Default: 0.1
        buffer_size_gb (float, optional): Size of audio to read into RAM in GB at any given time. Default: 10.0
            Note: This is the buffer size PER DataLoader worker. So total RAM = buffer_size_gb * num_workers
        buffer_reload_rate (int, optional): Number of items to generate before loading next chunk of dataset. Default: 10000
        half (bool, optional): Sotre audio samples as float 16. Default: False
        num_examples_per_epoch (int, optional): Define an epoch as certain number of audio examples. Default: 10000
        random_scale_input (bool, optional): Apply random gain scaling to input utterances. Default: False
        random_scale_target (bool, optional): Apply same random gain scaling to target utterances. Default: False
        augmentations (dict, optional): List of augmentation types to apply to inputs. Default: []
        freq_corrupt (bool, optional): Apply bad EQ filters. Default: False
        drc_corrupt (bool, optional): Apply an expander to corrupt dynamic range. Default: False
        ext (str, optional): Expected audio file extension. Default: "wav"
    """

    def __init__(
        self,
        audio_dir,
        input_dirs: List[str] = ["cleanraw"],
        subset: str = "train",
        length: int = 65536,
        train_frac: float = 0.8,
        val_per: float = 0.1,
        buffer_size_gb: float = 1.0,
        buffer_reload_rate: float = 1000,
        half: bool = False,
        num_examples_per_epoch: int = 10000,
        random_scale_input: bool = False,
        random_scale_target: bool = False,
        augmentations: dict = {},
        freq_corrupt: bool = False,
        drc_corrupt: bool = False,
        ext: str = "wav",
    ):
        super().__init__()
        self.audio_dir = audio_dir
        self.dataset_name = os.path.basename(audio_dir)
        self.input_dirs = input_dirs
        self.subset = subset
        self.length = length
        self.train_frac = train_frac
        self.val_per = val_per
        self.buffer_size_gb = buffer_size_gb
        self.buffer_reload_rate = buffer_reload_rate
        self.half = half
        self.num_examples_per_epoch = num_examples_per_epoch
        self.random_scale_input = random_scale_input
        self.random_scale_target = random_scale_target
        self.augmentations = augmentations
        self.freq_corrupt = freq_corrupt
        self.drc_corrupt = drc_corrupt
        self.ext = ext

        self.input_filepaths = []
        for input_dir in input_dirs:
            search_path = os.path.join(audio_dir, input_dir, f"*.{ext}")
            self.input_filepaths += glob.glob(search_path)
        self.input_filepaths = sorted(self.input_filepaths)

        # create dataset split based on subset
        self.input_filepaths = utils.split_dataset(
            self.input_filepaths,
            subset,
            train_frac,
        )

        # get details about input audio files
        input_files = {}
        input_dur_frames = 0
        for input_filepath in tqdm(self.input_filepaths, ncols=80):
            file_id = os.path.basename(input_filepath)
            audio_file = AudioFile(
                input_filepath,
                preload=False,
                half=half,
            )
            if audio_file.num_frames < (self.length * 2):
                continue
            input_files[file_id] = audio_file
            input_dur_frames += input_files[file_id].num_frames

        if len(list(input_files.items())) < 1:
            raise RuntimeError(f"No files found in {search_path}.")

        input_dur_hr = (input_dur_frames / input_files[file_id].sample_rate) / 3600
        print(
            f"\nLoaded {len(input_files)} files for {subset} = {input_dur_hr:0.2f} hours."
        )

        self.sample_rate = input_files[file_id].sample_rate

        # save a csv file with details about the train and test split
        splits_dir = os.path.join("configs", "splits")
        if not os.path.isdir(splits_dir):
            os.makedirs(splits_dir)
        csv_filepath = os.path.join(splits_dir, f"{self.dataset_name}_{self.subset}_set.csv")

        with open(csv_filepath, "w") as fp:
            dw = csv.DictWriter(fp, ["file_id", "filepath", "type", "subset"])
            dw.writeheader()
            for input_filepath in self.input_filepaths:
                dw.writerow(
                    {
                        "file_id": self.get_file_id(input_filepath),
                        "filepath": input_filepath,
                        "type": "input",
                        "subset": self.subset,
                    }
                )

        # some setup for iteratble loading of the dataset into RAM
        self.items_since_load = self.buffer_reload_rate

    def __len__(self):
        return self.num_examples_per_epoch

    def load_audio_buffer(self):
        self.input_files_loaded = {}  # clear audio buffer
        self.items_since_load = 0  # reset iteration counter
        nbytes_loaded = 0  # counter for data in RAM

        # different subset in each
        random.shuffle(self.input_filepaths)

        # load files into RAM
        for input_filepath in self.input_filepaths:
            file_id = os.path.basename(input_filepath)
            audio_file = AudioFile(
                input_filepath,
                preload=True,
                half=self.half,
            )

            if audio_file.num_frames < (self.length * 2):
                continue

            self.input_files_loaded[file_id] = audio_file

            nbytes = audio_file.audio.element_size() * audio_file.audio.nelement()
            nbytes_loaded += nbytes

            # check the size of loaded data
            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def generate_pair(self):
        # ------------------------ Input audio ----------------------
        rand_input_file_id = None
        input_file = None
        start_idx = None
        stop_idx = None
        while True:
            rand_input_file_id = self.get_random_file_id(self.input_files_loaded.keys())

            # use this random key to retrieve an input file
            input_file = self.input_files_loaded[rand_input_file_id]

            # load the audio data if needed
            if not input_file.loaded:
                raise RuntimeError("Audio not loaded.")

            # get a random patch of size `self.length` x 2
            start_idx, stop_idx = self.get_random_patch(
                input_file, int(self.length * 2)
            )
            if start_idx >= 0:
                break

        input_audio = input_file.audio[:, start_idx:stop_idx].clone().detach()
        input_audio = input_audio.view(1, -1)

        if self.half:
            input_audio = input_audio.float()

        # peak normalize to -12 dBFS
        input_audio /= input_audio.abs().max()
        input_audio *= 10 ** (-12.0 / 20)  # with min 3 dBFS headroom

        if len(list(self.augmentations.items())) > 0:
            if torch.rand(1).sum() < 0.5:
                input_audio_aug = augmentations.apply(
                    [input_audio],
                    self.sample_rate,
                    self.augmentations,
                )[0]
            else:
                input_audio_aug = input_audio.clone()
        else:
            input_audio_aug = input_audio.clone()

        input_audio_corrupt = input_audio_aug.clone()
        # apply frequency and dynamic range corrpution (expander)
        if self.freq_corrupt and torch.rand(1).sum() < 0.75:
            input_audio_corrupt = augmentations.frequency_corruption(
                [input_audio_corrupt], self.sample_rate
            )[0]

        # peak normalize again before passing through dynamic range expander
        input_audio_corrupt /= input_audio_corrupt.abs().max()
        input_audio_corrupt *= 10 ** (-12.0 / 20)  # with min 3 dBFS headroom

        if self.drc_corrupt and torch.rand(1).sum() < 0.10:
            input_audio_corrupt = augmentations.dynamic_range_corruption(
                [input_audio_corrupt], self.sample_rate
            )[0]

        # ------------------------ Target audio ----------------------
        # use the same augmented audio clip, add different random EQ and compressor

        target_audio_corrupt = input_audio_aug.clone()
        # apply frequency and dynamic range corrpution (expander)
        if self.freq_corrupt and torch.rand(1).sum() < 0.75:
            target_audio_corrupt = augmentations.frequency_corruption(
                [target_audio_corrupt], self.sample_rate
            )[0]

        # peak normalize again before passing through dynamic range compressor
        input_audio_corrupt /= input_audio_corrupt.abs().max()
        input_audio_corrupt *= 10 ** (-12.0 / 20)  # with min 3 dBFS headroom

        if self.drc_corrupt and torch.rand(1).sum() < 0.75:
            target_audio_corrupt = augmentations.dynamic_range_compression(
                [target_audio_corrupt], self.sample_rate
            )[0]

        return input_audio_corrupt, target_audio_corrupt

    def __getitem__(self, _):
        """ """

        # increment counter
        self.items_since_load += 1

        # load next chunk into buffer if needed
        if self.items_since_load > self.buffer_reload_rate:
            self.load_audio_buffer()

        # generate pairs for style training
        input_audio, target_audio = self.generate_pair()

        # ------------------------ Conform length of files -------------------
        input_audio = utils.conform_length(input_audio, int(self.length * 2))
        target_audio = utils.conform_length(target_audio, int(self.length * 2))

        # ------------------------ Apply fade in and fade out -------------------
        input_audio = utils.linear_fade(input_audio, sample_rate=self.sample_rate)
        target_audio = utils.linear_fade(target_audio, sample_rate=self.sample_rate)

        # ------------------------ Final normalizeation ----------------------
        # always peak normalize final input to -12 dBFS
        input_audio /= input_audio.abs().max()
        input_audio *= 10 ** (-12.0 / 20.0)

        # always peak normalize the target to -12 dBFS
        target_audio /= target_audio.abs().max()
        target_audio *= 10 ** (-12.0 / 20.0)

        return input_audio, target_audio

    @staticmethod
    def get_random_file_id(keys):
        # generate a random index into the keys of the input files
        rand_input_idx = torch.randint(0, len(keys) - 1, [1])[0]
        # find the key (file_id) correponding to the random index
        rand_input_file_id = list(keys)[rand_input_idx]

        return rand_input_file_id

    @staticmethod
    def get_random_patch(audio_file, length, check_silence=True):
        silent = True
        count = 0
        while silent:
            count += 1
            start_idx = torch.randint(0, audio_file.num_frames - length - 1, [1])[0]
            # int(torch.rand(1) * (audio_file.num_frames - length))
            stop_idx = start_idx + length
            patch = audio_file.audio[:, start_idx:stop_idx].clone().detach()

            length = patch.shape[-1]
            first_patch = patch[..., : length // 2]
            second_patch = patch[..., length // 2 :]

            if (
                (first_patch**2).mean() > 1e-5 and (second_patch**2).mean() > 1e-5
            ) or not check_silence:
                silent = False

            if count > 100:
                print("get_random_patch count", count)
                return -1, -1
                # break

        return start_idx, stop_idx

    def get_file_id(self, filepath):
        """Given a filepath extract the DAPS file id.

        Args:
            filepath (str): Path to an audio files in the DAPS dataset.

        Returns:
            file_id (str): DAPS file id of the form <participant_id>_<script_id>
            file_set (str): The DAPS set to which the file belongs.
        """
        file_id = os.path.basename(filepath).split("_")[:2]
        file_id = "_".join(file_id)
        return file_id

    def get_file_set(self, filepath):
        """Given a filepath extract the DAPS file set name.

        Args:
            filepath (str): Path to an audio files in the DAPS dataset.

        Returns:
            file_set (str): The DAPS set to which the file belongs.
        """
        file_set = os.path.basename(filepath).split("_")[2:]
        file_set = "_".join(file_set)
        file_set = file_set.replace(f".{self.ext}", "")
        return file_set
