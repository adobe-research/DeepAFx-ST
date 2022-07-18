import os
import glob
import torch
import torchaudio
from tqdm import tqdm


class StyleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_dir: str,
        subset: str = "train",
        sample_rate: int = 24000,
        length: int = 131072,
    ) -> None:
        super().__init__()
        self.audio_dir = audio_dir
        self.subset = subset
        self.sample_rate = sample_rate
        self.length = length

        self.style_dirs = glob.glob(os.path.join(audio_dir, subset, "*"))
        self.style_dirs = [sd for sd in self.style_dirs if os.path.isdir(sd)]
        self.num_classes = len(self.style_dirs)
        self.class_labels = {"broadcast" : 0, "telephone": 1, "neutral": 2, "bright": 3, "warm": 4}

        self.examples = []
        for n, style_dir in enumerate(self.style_dirs):

            # get all files in style dir
            style_filepaths = glob.glob(os.path.join(style_dir, "*.wav"))
            style_name = os.path.basename(style_dir)
            for style_filepath in tqdm(style_filepaths, ncols=120):
                # load audio file
                x, sr = torchaudio.load(style_filepath)

                # sum to mono if needed
                if x.shape[0] > 1:
                    x = x.mean(dim=0, keepdim=True)

                # resample
                if sr != self.sample_rate:
                    x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)

                # crop length after resample
                if x.shape[-1] >= self.length:
                    x = x[...,:self.length]

                # store example
                example = (x, self.class_labels[style_name])
                self.examples.append(example)

        print(f"Loaded {len(self.examples)} examples for {subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        x = example[0]
        y = example[1]
        return x, y
