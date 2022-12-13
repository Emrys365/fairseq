# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import joblib

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_feature_shard(feat_dir, split, nshard, rank, percent):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    if percent < 0:
        return np.load(feat_path, mmap_mode="r")
    else:
        nsample = int(np.ceil(len(lengs) * percent))
        indices = np.random.choice(len(lengs), nsample, replace=False)
        feat = np.load(feat_path, mmap_mode="r")
        sampled_feat = np.concatenate(
            [feat[offsets[i] : offsets[i] + lengs[i]] for i in indices], axis=0
        )
        logger.info(
            (
                f"sampled {nsample} utterances, {len(sampled_feat)} frames "
                f"from shard {rank}/{nshard}"
            )
        )
        return sampled_feat


def load_feature(feat_dir, split, nshard, seed, percent):
    assert percent <= 1.0
    feat = np.concatenate(
        [
            load_feature_shard(feat_dir, split, nshard, r, percent)
            for r in range(nshard)
        ],
        axis=0,
    )
    logging.info(f"loaded feature with dimension {feat.shape}")
    return feat


def learn_kmeans(
    feat_dir,
    split,
    nshard,
    km_path,
    n_clusters,
    seed,
    percent,
    init,
    max_iter,
    batch_size,
    tol,
    n_init,
    reassignment_ratio,
    max_no_improvement,
):
    np.random.seed(seed)
    feat = load_feature(feat_dir, split, nshard, seed, percent)
    km_model = get_km_model(
        n_clusters,
        init,
        max_iter,
        batch_size,
        tol,
        max_no_improvement,
        n_init,
        reassignment_ratio,
    )
    km_model.fit(feat)
    joblib.dump(km_model, km_path)

    inertia = -km_model.score(feat) / len(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "feat_dir", type=str, help="Path to the directory containing extracted features"
    )
    parser.add_argument(
        "split",
        type=str,
        help="Name of one tsv file (without .tsv suffix) with the first row containing "
        "the root directory of audios and the rest rows listing each audio path",
    )
    parser.add_argument(
        "n_clusters", type=int, help="Number of cluster centroids for K-Means"
    )
    parser.add_argument(
        "--km_path",
        type=str,
        default="km_model",
        help="Output directory for storing the K-Means model",
    )
    parser.add_argument(
        "--nshard",
        type=int,
        default=1,
        help="Partition of data in this process (used for multiprocessing)\n"
        "This would load all `nshard` parts of the extracted features",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--percent",
        default=-1,
        type=float,
        help="Sample a subset with the specified percentage; -1 for all",
    )
    parser.add_argument(
        "--init",
        default="k-means++",
        type=str,
        help="Initialization method for K-Means",
    )
    parser.add_argument(
        "--max_iter",
        default=100,
        type=int,
        help="Maximum number of iterations over the complete dataset before "
        "stopping independently of any early stopping criterion heuristics",
    )
    parser.add_argument(
        "--batch_size",
        default=10000,
        type=int,
        help="Size of the mini batches in K-Means",
    )
    parser.add_argument(
        "--tol",
        default=0.0,
        type=float,
        help="Control early stopping based on the relative center changes as "
        "measured by a smoothed, variance-normalized of the mean center squared "
        "position changes (See sklearn.cluster.MiniBatchKMeans for more details)",
    )
    parser.add_argument(
        "--max_no_improvement",
        default=100,
        type=int,
        help="Control early stopping based on the consecutive number of mini "
        "batches that does not yield an improvement on the smoothed inertia",
    )
    parser.add_argument(
        "--n_init",
        default=20,
        type=int,
        help="Number of random initializations that are tried",
    )
    parser.add_argument(
        "--reassignment_ratio",
        default=0.0,
        type=float,
        help="Control the fraction of the maximum number of counts for a center "
        "to be reassigned. A higher value means that low count centers are more "
        "easily reassigned, which means that the model will take longer to "
        "converge, but should converge in a better clustering.",
    )
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(**vars(args))
