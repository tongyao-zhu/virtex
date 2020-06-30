import os
import random
from typing import Callable, List

import albumentations as alb
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from virtex.data.readers import LmdbReader
from virtex.data.structures import ImageCaptionInstance, ImageCaptionBatch
from virtex.data.tokenizers import SentencePieceBPETokenizer
from virtex.data import transforms as T


class VideoCaptioningDataset(Dataset):
    def __init__(
            self,
            data_root: str,
            csv: str,
            split: str,
            tokenizer: SentencePieceBPETokenizer,
            image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
            padded_length: int = 256,
            max_caption_length: int = 50,
            use_single_caption: bool = False,
            percentage: float = 100.0,
    ):
        self.data_root = data_root
        self.padded_length = padded_length
        info_df = pd.read_csv(os.path.join(data_root, csv), delimiter="|")
        self.video_list = []
        for index, row in info_df.iterrows():
            self.video_list.append((index, row['name'], [row['orth']]))
        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.use_single_caption = use_single_caption
        self.padding_idx = tokenizer.token_to_id("<unk>")

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx: int) -> ImageCaptionInstance:

        video_id, video_name, captions = self.video_list[idx]
        video = np.load(os.path.join(self.data_root, "video_vectors", f"{video_name}.npy"))
        # Pick a random caption or first caption and process (transform) it.
        if self.use_single_caption:
            caption = captions[0]
        else:
            caption = random.choice(captions)

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        #print("padded length is {}".format(self.padded_length))
        padded_video = np.zeros([self.padded_length, 224, 224, 3])
        # perform downsampling:
        if len(video) > self.padded_length:
            indices_list = [int(len(video)/self.padded_length * x) for x in range(self.padded_length)]
            while (indices_list[-1] >= len(video)):
                indices_list[-1] -=1
            video = video[indices_list]
        for i in range(min(len(video), self.padded_length)):
            image_caption = self.image_transform(image=video[i], caption=caption)
            image, caption = image_caption['image'], image_caption['caption']
            padded_video[i] = image
        processed_video = np.transpose(padded_video, (0, 3, 1, 2))

        caption_tokens = self.caption_transform(caption=caption)["caption"]
        return ImageCaptionInstance(video_id, processed_video, caption_tokens)

    def collate_fn(self, instances: List[ImageCaptionInstance]) -> ImageCaptionBatch:
        return ImageCaptionBatch(instances, padding_value=self.padding_idx)


class CaptioningDataset(Dataset):
    r"""
    A dataset which provides image-caption (forward and backward) pairs from
    a serialized LMDB file (COCO Captions in this codebase). This is used for
    pretraining tasks which use captions - bicaptioning, forward captioning and
    token classification.

    This dataset also supports training on a randomly selected subset of the
    full dataset.

    Parameters
    ----------
    data_root: str, optional (default = "datasets/coco")
        Path to the dataset root directory. This must contain the serialized
        LMDB files (for COCO ``train2017`` and ``val2017`` splits).
    split: str, optional (default = "train")
        Which split (from COCO 2017 version) to read. One of ``{"train", "val"}``.
    tokenizer: virtex.data.tokenizers.SentencePieceBPETokenizer
        A tokenizer which has the mapping between word tokens and their
        integer IDs.
    image_tranform: Callable, optional (default = virtex.data.transforms.DEFAULT_IMAGE_TRANSFORM)
        A list of transformations, from either `albumentations
        <https://albumentations.readthedocs.io/en/latest/>`_ or :mod:`virtex.data.transforms`
        to be applied on the image.
    max_caption_length: int, optional (default = 30)
        Maximum number of tokens to keep in output caption tokens. Extra tokens
        will be trimmed from the right end of the token list.
    use_single_caption: bool, optional (default = False)
        COCO Captions provides five captions per image. If this is True, only
        one fixed caption per image is use fo training (used for an ablation).
    percentage: float, optional (default = 100.0)
        Randomly sample this much percentage of full dataset for training.
    """

    def __init__(
            self,
            data_root: str,
            split: str,
            tokenizer: SentencePieceBPETokenizer,
            image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
            max_caption_length: int = 30,
            use_single_caption: bool = False,
            percentage: float = 100.0,
    ):
        lmdb_path = os.path.join(data_root, f"serialized_{split}.lmdb")
        self.reader = LmdbReader(lmdb_path, percentage=percentage)

        self.image_transform = image_transform
        self.caption_transform = alb.Compose(
            [
                T.NormalizeCaption(),
                T.TokenizeCaption(tokenizer),
                T.TruncateCaptionTokens(max_caption_length),
            ]
        )
        self.use_single_caption = use_single_caption
        self.padding_idx = tokenizer.token_to_id("<unk>")

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx: int) -> ImageCaptionInstance:

        image_id, image, captions = self.reader[idx]

        # Pick a random caption or first caption and process (transform) it.
        if self.use_single_caption:
            caption = captions[0]
        else:
            caption = random.choice(captions)

        # Transform image-caption pair and convert image from HWC to CHW format.
        # Pass in caption to image_transform due to paired horizontal flip.
        # Caption won't be tokenized/processed here.
        image_caption = self.image_transform(image=image, caption=caption)
        image, caption = image_caption["image"], image_caption["caption"]
        image = np.transpose(image, (2, 0, 1))

        caption_tokens = self.caption_transform(caption=caption)["caption"]
        return ImageCaptionInstance(image_id, image, caption_tokens)

    def collate_fn(self, instances: List[ImageCaptionInstance]) -> ImageCaptionBatch:
        return ImageCaptionBatch(instances, padding_value=self.padding_idx)
