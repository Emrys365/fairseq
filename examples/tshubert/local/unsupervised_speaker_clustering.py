#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from tqdm.contrib.concurrent import process_map  # or thread_map
import pickle
import faiss
from pathlib import Path


def worker(line):
    key, vec = line
    vec = np.load(vec)
    return vec


def read_data(f_path, max_workers=8, max_chunksize=1000):
    print("Reading data start ......")
    pkl_dir = Path(f_path).parent / "embs.pkl"
    if pkl_dir.exists():
        return pkl_dir

    has_uid = None
    lines = []
    with open(f_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if has_uid is None:
                has_uid = len(line.strip().split()) > 1
            if has_uid:
                uid, path = line.strip().split(maxsplit=1)
            else:
                uid = i
                path = line.strip()
            lines.append((uid, path))
    ret = process_map(
        worker,
        lines,
        max_workers=max_workers,
        chunksize=max_chunksize,
    )
    all_data = np.stack(ret)
    with open(pkl_dir, "wb") as f:
        pickle.dump(all_data, f)
    print("Reading data done!")
    return pkl_dir


def load_data(pkl_dir):
    with open(pkl_dir, "rb") as f:
        all_data = pickle.load(f)
    print(all_data.shape)
    return all_data


def preprocess_data(data, norm=False, pca=None, avg=False):
    _, ndim = data.shape
    data = data.astype("float32")
    if pca:
        mat = faiss.PCAMatrix(ndim, pca, eigen_power=-0.5)
        mat.train(data)
        assert mat.is_trained
        data = mat.apply_py(data)
    if avg:
        means = np.mean(data, axis=0)
        data = data - means
    if norm:
        row_sums = np.linalg.norm(data, axis=1)
        data = data / row_sums[:, np.newaxis]
    return data


def run_clustering(data, num_clusters, f_name, verbose=False):
    n_data, d = data.shape

    # Installation of faiss-gpu is required for gpu=True
    clus = faiss.Kmeans(
        d,
        num_clusters,
        niter=20,
        verbose=verbose,
        seed=1024,
        max_points_per_centroid=1000000,
        gpu=True,
    )
    clus.train(data)

    D, I = clus.index.search(data, 1)
    write(D, I, f_name)

    inertia = inertia_dense(data, clus.centroids, I)
    return D, I, inertia


def run_ahc_clustering(data, num_clusters, f_name, verbose=False, tp=0):
    n_data, d = data.shape

    # iteration 1
    clus = faiss.Kmeans(
        d,
        num_clusters,
        niter=20,
        verbose=verbose,
        seed=1024,
        max_points_per_centroid=1000000,
        gpu=1,
    )
    clus.train(data)

    # iteration 2
    D, I = clus.index.search(data, 1)

    center = clus.centroids

    clus2 = faiss.Kmeans(
        d,
        2000,
        niter=20,
        verbose=verbose,
        seed=1024,
        max_points_per_centroid=1000000,
        gpu=1,
    )
    clus2.train(center)
    if tp == 0:
        D2, I2 = clus2.index.search(data, 1)
        write(D2, I2, f_name)
    elif tp == 1:
        D2, I2 = clus2.index.search(center, 1)

    inertia = inertia_dense(data, clus.centroids, I2)
    return D2, I2, inertia


def write(D, I, f_name):
    tmp_data = []
    for i in range(len(D)):
        # distance, label
        tmp_data.append([D[i], I[i]])
    # tmp_data = sorted(tmp_data, key=lambda student: student[1])
    with open(f_name, "w") as f:
        for i in range(len(tmp_data)):
            f.write(
                # str(np.squeeze(tmp_data[i][1]))
                # + " "
                +str(np.squeeze(tmp_data[i][2]))
                + "\n"
            )
    print("write_done")


def inertia_dense(data, centroids, labels):
    """Compute inertia for dense (rather than sparse) input data

    Sum of squared distance between each sample and its assigned center.
    """
    labels = np.squeeze(labels)
    assert len(data) == len(labels), (len(data), len(labels))
    assert min(labels) == 0 and max(labels) == len(centroids) - 1, (
        min(labels),
        max(labels),
        len(centroids),
    )

    centroids = np.take(centroids, labels, axis=0)
    assert data.shape == centroids.shape, (data.shape, centroids.shape)
    inertia = np.square(data - centroids).sum()
    return inertia


def find_elbow(x, y):
    """Find the elbow automatically

    Ref: https://stackoverflow.com/a/49807209
    """
    from kneed import KneeLocator

    # curve="concave" -> detect knees. curve="convex" -> detect elbows.

    # Using interp_method="polynomial" will modify the original data `y` via `np.polyfit`
    # kn = KneeLocator(
    #     x, y, curve="convex", direction="decreasing", interp_method="polynomial", polynomial_degree=7
    # )

    # Using interp_method="interp1d" will not modify the original data `y`
    kn = KneeLocator(
        x, y, curve="convex", direction="decreasing", interp_method="interp1d"
    )
    return kn.knee


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scp",
        type=str,
        help="Path to the scp file containing paths of speaker embeddings",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Path to the output file containing clustering labels",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    f_path = args.scp
    pkl_dir = read_data(f_path, max_workers=8)  # only need to run this once
    data = load_data(pkl_dir)
    data = preprocess_data(data, norm=True, pca=None, avg=True)

    nclusters, values = [], []
    # test #clusters in the range [10, 20, 30, ..., max_num_clusters]
    max_num_clusters = min(10000, len(data) // 10 * 10)
    for n in range(10, max_num_clusters + 10, 10):
        if args.outfile:
            outfile = args.outfile
        else:
            outfile = Path(f_path).parent / f"resnet293_{n}_cluster_norm.txt"
            # outfile = Path(f_path).parent / f"ecapa_big_dino_{n}_cluster_norm.txt"
        D, I, inertia = run_clustering(
            data, num_clusters=n, f_name=outfile, verbose=args.verbose
        )
        # D, I, inertia = run_ahc_clustering(
        #     data, num_clusters=n, f_name=outfile, verbose=args.verbose
        # )
        print(f"[{n} clusters] inertia={inertia}")

        nclusters.append(n)
        values.append(inertia)
        plt.clf()
        plt.plot(nclusters, values)
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.grid()
        plt.savefig("inertia.png")

    elbow = find_elbow(nclusters, values)
    print(f"Elbow is at nclusters={elbow}")
    with open("inertia.csv", "w") as f:
        for n, inertia in zip(nclusters, values):
            f.write(f"{n},{inertia}\n")
