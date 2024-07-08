# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import joblib
import numpy as np
import torch
import tqdm

from tshubert_extract_kmeans_features import TsHuBERTFeatureReader
from feature_utils import get_shard_range, get_path_iterator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x**2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_enroll_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        def iterate():
            for line in lines:
                yield line
    return iterate, len(lines)


def get_feat_iterator(iter1, iter2, reader, enroll_len=None):
    def iterate():
        for (path, nsample), path_enroll in zip(iter1, iter2):
            feat = reader.get_feats(path, nsample, path_enroll, enroll_len)
            yield feat.cpu().numpy()

    return iterate


def dump_label(
    tsv_dir,
    split,
    km_path,
    ckpt_path,
    layer,
    nshard,
    rank,
    lab_dir,
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
    iterator = get_feat_iterator(generator(), generator_enroll(), reader, enroll_len)

    apply_kmeans = ApplyKmeans(km_path)

    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km"
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator(), total=num):
            # feat = torch.from_numpy(feat).cuda()
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tsv_dir", type=str, help="Path to the directory containing tsv files"
    )
    parser.add_argument(
        "split",
        type=str,
        help="Name of one tsv file (without .tsv suffix) with the first row containing "
        "the root directory of audios and the rest rows listing each audio path",
    )
    parser.add_argument(
        "km_path", type=str, help="Path to the pretrained K-Means model"
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
        "This would process the `rank`-th part of the sharded feature files",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Rank index of the current process, which must be an integer in "
        "range [1, nshard]. (used for multiprocessing)",
    )
    parser.add_argument(
        "--lab_dir", type=str, help="Output directory for storing K-Means labels"
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
    logging.info(str(args))
    np.random.seed(args.seed)

    del args.seed
    dump_label(**vars(args))
