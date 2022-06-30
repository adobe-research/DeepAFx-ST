import os
import glob
import torch
import warnings
import torchaudio
import pyloudnorm as pyln


class AudioFile(object):
    def __init__(self, filepath, preload=False, half=False, target_loudness=None):
        """Base class for audio files to handle metadata and loading.

        Args:
            filepath (str): Path to audio file to load from disk.
            preload (bool, optional): If set, load audio data into RAM. Default: False
            half (bool, optional): If set, store audio data as float16 to save space. Default: False
            target_loudness (float, optional): Loudness normalize to dB LUFS value. Default:
        """
        super().__init__()

        self.filepath = filepath
        self.half = half
        self.target_loudness = target_loudness
        self.loaded = False

        if preload:
            self.load()
            num_frames = self.audio.shape[-1]
            num_channels = self.audio.shape[0]
        else:
            metadata = torchaudio.info(filepath)
            audio = None
            self.sample_rate = metadata.sample_rate
            num_frames = metadata.num_frames
            num_channels = metadata.num_channels

        self.num_frames = num_frames
        self.num_channels = num_channels

    def load(self):
        audio, sr = torchaudio.load(self.filepath, normalize=True)
        self.audio = audio
        self.sample_rate = sr

        if self.target_loudness is not None:
            self.loudness_normalize()

        if self.half:
            self.audio = audio.half()

        self.loaded = True

    def loudness_normalize(self):
        meter = pyln.Meter(self.sample_rate)

        # conver mono to stereo
        if self.audio.shape[0] == 1:
            tmp_audio = self.audio.repeat(2, 1)
        else:
            tmp_audio = self.audio

        # measure integrated loudness
        input_loudness = meter.integrated_loudness(tmp_audio.numpy().T)

        # compute and apply gain
        gain_dB = self.target_loudness - input_loudness
        gain_ln = 10 ** (gain_dB / 20.0)
        self.audio *= gain_ln

        # check for potentially clipped samples
        if self.audio.abs().max() >= 1.0:
            warnings.warn("Possible clipped samples in output.")


class AudioFileDataset(torch.utils.data.Dataset):
    """Base class for audio file datasets loaded from disk.

    Datasets can be either paired or unpaired. A paired dataset requires passing the `target_dir` path.

    Args:
        input_dir (List[str]): List of paths to the directories containing input audio files.
        target_dir (List[str], optional): List of paths to the directories containing correponding audio files. Default: []
        subset (str, optional): Dataset subset. One of ["train", "val", "test"]. Default: "train"
        length (int, optional): Number of samples to load for each example. Default: 65536
        normalize (bool, optional): Normalize audio amplitiude to -1 to 1. Default: True
        train_frac (float, optional): Fraction of the files to use for training subset. Default: 0.8
        val_frac (float, optional): Fraction of the files to use for validation subset. Default: 0.1
        preload (bool, optional): Read audio files into RAM at the start of training. Default: False
        num_examples_per_epoch (int, optional): Define an epoch as certain number of audio examples. Default: 10000
        ext (str, optional): Expected audio file extension. Default: "wav"
    """

    def __init__(
        self,
        input_dirs,
        target_dirs=[],
        subset="train",
        length=65536,
        normalize=True,
        train_per=0.8,
        val_per=0.1,
        preload=False,
        num_examples_per_epoch=10000,
        ext="wav",
    ):
        super().__init__()
        self.input_dirs = input_dirs
        self.target_dirs = target_dirs
        self.subset = subset
        self.length = length
        self.normalize = normalize
        self.train_per = train_per
        self.val_per = val_per
        self.preload = preload
        self.num_examples_per_epoch = num_examples_per_epoch
        self.ext = ext

        self.input_filepaths = []
        for input_dir in input_dirs:
            search_path = os.path.join(input_dir, f"*.{ext}")
            self.input_filepaths += glob.glob(search_path)
        self.input_filepaths = sorted(self.input_filepaths)

        self.target_filepaths = []
        for target_dir in target_dirs:
            search_path = os.path.join(target_dir, f"*.{ext}")
            self.target_filepaths += glob.glob(search_path)
        self.target_filepaths = sorted(self.target_filepaths)

        # both sets must have same number of files in paired dataset
        assert len(self.target_filepaths) == len(self.input_filepaths)

        # get details about audio files
        self.input_files = []
        for input_filepath in self.input_filepaths:
            self.input_files.append(
                AudioFile(input_filepath, preload=preload, normalize=normalize)
            )

        self.target_files = []
        if target_dir is not None:
            for target_filepath in self.target_filepaths:
                self.target_files.append(
                    AudioFile(target_filepath, preload=preload, normalize=normalize)
                )

    def __len__(self):
        return self.num_examples_per_epoch

    def __getitem__(self, idx):
        """ """

        # index the current audio file
        input_file = self.input_files[idx]

        # load the audio data if needed
        if not input_file.loaded:
            input_file.load()

        # get a random patch of size `self.length`
        start_idx = int(torch.rand() * (input_file.num_frames - self.length))
        stop_idx = start_idx + self.length
        input_audio = input_file.audio[:, start_idx:stop_idx]

        # if there is a target file, get it (and load)
        if len(self.target_files) > 0:
            target_file = self.target_files[idx]

            if not target_file.loaded:
                target_file.load()

            # use the same cropping indices
            target_audio = target_file.audio[:, start_idx:stop_idx]

            return input_audio, target_audio
        else:
            return input_audio
