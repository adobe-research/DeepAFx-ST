import os
import json
import glob
import torch
import random
from tqdm import tqdm

# from deepafx_st.plugins.channel import Channel
from deepafx_st.processors.processor import Processor
from deepafx_st.data.audio import AudioFile
import deepafx_st.utils as utils


class DSPProxyDataset(torch.utils.data.Dataset):
    """Class for generating input-output audio from Python DSP effects.

    Args:
        input_dir (List[str]): List of paths to the directories containing input audio files.
        processor (Processor): Processor object to create proxy of.
        processor_type (str): Processor name.
        subset (str, optional): Dataset subset. One of ["train", "val", "test"]. Default: "train"
        buffer_size_gb (float, optional): Size of audio to read into RAM in GB at any given time. Default: 10.0
            Note: This is the buffer size PER DataLoader worker. So total RAM = buffer_size_gb * num_workers
        buffer_reload_rate (int, optional): Number of items to generate before loading next chunk of dataset. Default: 10000
        length (int, optional): Number of samples to load for each example. Default: 65536
        num_examples_per_epoch (int, optional): Define an epoch as certain number of audio examples. Default: 10000
        ext (str, optional): Expected audio file extension. Default: "wav"
        hard_clip (bool, optional): Hard clip outputs between -1 and 1. Default: True
    """

    def __init__(
        self,
        input_dir: str,
        processor: Processor,
        processor_type: str,
        subset="train",
        length=65536,
        buffer_size_gb=1.0,
        buffer_reload_rate=1000,
        half=False,
        num_examples_per_epoch=10000,
        ext="wav",
        soft_clip=True,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.processor = processor
        self.processor_type = processor_type
        self.subset = subset
        self.length = length
        self.buffer_size_gb = buffer_size_gb
        self.buffer_reload_rate = buffer_reload_rate
        self.half = half
        self.num_examples_per_epoch = num_examples_per_epoch
        self.ext = ext
        self.soft_clip = soft_clip

        search_path = os.path.join(input_dir, f"*.{ext}")
        self.input_filepaths = glob.glob(search_path)
        self.input_filepaths = sorted(self.input_filepaths)

        if len(self.input_filepaths) < 1:
            raise RuntimeError(f"No files found in {input_dir}.")

        # get training split
        self.input_filepaths = utils.split_dataset(
            self.input_filepaths, self.subset, 0.9
        )

        # get details about audio files
        cnt = 0
        self.input_files = {}
        for input_filepath in tqdm(self.input_filepaths, ncols=80):
            file_id = os.path.basename(input_filepath)
            audio_file = AudioFile(
                input_filepath,
                preload=False,
                half=half,
            )
            if audio_file.num_frames < self.length:
                continue
            self.input_files[file_id] = audio_file
            self.sample_rate = self.input_files[file_id].sample_rate
            cnt += 1
            if cnt > 1000:
                break

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

            if audio_file.num_frames < self.length:
                continue

            self.input_files_loaded[file_id] = audio_file

            nbytes = audio_file.audio.element_size() * audio_file.audio.nelement()
            nbytes_loaded += nbytes

            if nbytes_loaded > self.buffer_size_gb * 1e9:
                break

    def __getitem__(self, _):
        """ """

        # increment counter
        self.items_since_load += 1

        # load next chunk into buffer if needed
        if self.items_since_load > self.buffer_reload_rate:
            self.load_audio_buffer()

        rand_input_file_id = utils.get_random_file_id(self.input_files_loaded.keys())
        # use this random key to retrieve an input file
        input_file = self.input_files_loaded[rand_input_file_id]

        # load the audio data if needed
        if not input_file.loaded:
            input_file.load()

        # get a random patch of size `self.length`
        # start_idx, stop_idx = utils.get_random_patch(input_file, self.sample_rate, self.length)
        start_idx, stop_idx = utils.get_random_patch(input_file, self.length)
        input_audio = input_file.audio[:, start_idx:stop_idx].clone().detach()

        # random scaling
        input_audio /= input_audio.abs().max()
        scale_dB = (torch.rand(1).squeeze().numpy() * 12) + 12
        input_audio *= 10 ** (-scale_dB / 20.0)

        # generate random parameters (uniform) over 0 to 1
        params = torch.rand(self.processor.num_control_params)

        # expects batch dim
        # apply plugins with random parameters
        if self.processor_type == "channel":
            params[-1] = 0.5  # set makeup gain to 0dB
            target_audio = self.processor(
                input_audio.view(1, 1, -1),
                params.view(1, -1),
            )
            target_audio = target_audio.view(1, -1)
        elif self.processor_type == "peq":
            target_audio = self.processor(
                input_audio.view(1, 1, -1).numpy(),
                params.view(1, -1).numpy(),
            )
            target_audio = torch.tensor(target_audio).view(1, -1)
        elif self.processor_type == "comp":
            params[-1] = 0.5  # set makeup gain to 0dB
            target_audio = self.processor(
                input_audio.view(1, 1, -1).numpy(),
                params.view(1, -1).numpy(),
            )
            target_audio = torch.tensor(target_audio).view(1, -1)

        # clip
        if self.soft_clip:
            # target_audio = target_audio.clamp(-2.0, 2.0)
            target_audio = torch.tanh(target_audio / 2.0) * 2.0

        return input_audio, target_audio, params
