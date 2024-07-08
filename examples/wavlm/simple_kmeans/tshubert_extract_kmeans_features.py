# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import fairseq
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from npy_append_array import NpyAppendArray
from tqdm import tqdm

from feature_utils import get_path_iterator, get_shard_range


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_wavlm_feature")


def dump_tshubert_feature(
    reader, generator, generator_enroll, num, enroll_len, split, nshard, rank, feat_dir
):
    iterator = generator()
    iterator_enroll = generator_enroll()

    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for (path, nsample), path_enroll in tqdm(
            zip(iterator, iterator_enroll), total=num
        ):
            feat = reader.get_feats(path, nsample, path_enroll, enroll_len)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


def get_enroll_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                yield line
    return iterate, len(lines)


class TsHuBERTFeatureReader(object):
    def __init__(self, ckpt_path, layer, max_chunk=1600000, device="cpu"):
        self.device = device
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
        self.model = model[0].eval()
        if device == "gpu":
            self.model.cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def crop_audio_to_length(self, audio, audio_size):
        size = len(audio)
        diff = size - audio_size
        if diff <= 0:
            return audio
        else:
            start, end = 0, audio_size
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
            return audio[start:end]

    def get_feats(self, path, ref_len=None, path_enroll=None, enroll_len=None):
        x = self.read_audio(path, ref_len)
        assert path_enroll is not None
        enrollment = self.read_audio(path_enroll)
        if enroll_len is not None:
            enrollment = self.crop_audio_to_length(enrollment, enroll_len)
        with torch.no_grad():
            if self.device == "gpu":
                x = torch.from_numpy(x).float().cuda()
                enroll = torch.from_numpy(enrollment).float().cuda()
            else:
                x = torch.from_numpy(x).float()
                enroll = torch.from_numpy(enrollment).float()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
                enroll = F.layer_norm(enroll, enroll.shape)
            x = x.view(1, -1)
            enroll = enroll.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    enrollment=enroll,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def main(
    tsv_dir,
    split,
    ckpt_path,
    layer,
    nshard,
    rank,
    feat_dir,
    max_chunk,
    device,
    enroll_len,
):
    rank = rank - 1
    reader = TsHuBERTFeatureReader(ckpt_path, layer, max_chunk, device=device)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    generator_enroll, num2 = get_enroll_iterator(
        f"{tsv_dir}/{split}.enroll", nshard, rank
    )
    assert num == num2, (num, num2)
    dump_tshubert_feature(
        reader,
        generator,
        generator_enroll,
        num,
        enroll_len,
        split,
        nshard,
        rank,
        feat_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tsv_dir",
        type=str,
        help="Path to the directory containing tsv files",
    )
    parser.add_argument(
        "split",
        type=str,
        help="Name of one tsv file (without .tsv suffix) with the first row containing "
        "the root directory of audios and the rest rows listing each audio path",
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument(
        "--layer",
        type=int,
        default=8,
        help="Index of the network layer for outputting features",
    )
    parser.add_argument(
        "--nshard",
        type=int,
        default=1,
        help="Partition of data in this process (used for multiprocessing)\n"
        "This would shard the tsv file into `nshard` parts and extract features "
        "for the `rank`-th shard",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Rank index of the current process, which must be an integer in "
        " range [1, nshard]. (used for multiprocessing)",
    )
    parser.add_argument(
        "--feat_dir",
        type=str,
        default="feats",
        help="Path to the output directory for storing extracted features",
    )
    parser.add_argument(
        "--max_chunk",
        type=int,
        default=1600000,
        help="Chunk size used when processing long-form input speech"
        "(If out-of-memory, consider reduce this value)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "gpu"),
        help="'cuda' or 'cpu device",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--enroll_len",
        type=int,
        default=None,
        help="If not None, the enrollment will be chomped to the specified length",
    )
    args = parser.parse_args()
    logger.info(args)
    np.random.seed(args.seed)

    del args.seed
    # extracted feature: (T, D)
    main(**vars(args))
