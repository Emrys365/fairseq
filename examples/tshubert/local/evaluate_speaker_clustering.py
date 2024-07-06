#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    fowlkes_mallows_score,
    homogeneity_score,
    mutual_info_score,
    normalized_mutual_info_score,
    rand_score,
    silhouette_score,
    v_measure_score,
)


def calculate_metrics(labels_true, labels_pred):
    """Calculates the clutering measures for the given labels.

    Args:
        labels_true (list): The ground-truth labels. (n_samples,)
        labels_pred (list): The predicted labels. (n_samples,)
    Returns:
        dict: A dictionary containing the clustering measures.
    """
    return {
        # https://scikit-learn.org/stable/modules/clustering.html#rand-index
        # Similarity score between 0.0 and 1.0, inclusive, 1.0 stands for perfect match.
        # RI = (number of agreeing pairs) / (number of pairs)
        "rand_index": rand_score(labels_true, labels_pred),  # symmetric
        "adjusted_rand_index": adjusted_rand_score(labels_true, labels_pred),  # symmetric
        # https://scikit-learn.org/stable/modules/clustering.html#mutual-information-based-scores
        # Perfect labeling is scored 1.0 for normalized_mutual_info and adjusted_mutual_info.
        # Values close to zero indicate two label assignments that are largely independent,
        #   while values close to one indicate significant agreement.
        # Further, an AMI of exactly 1 indicates that the two label assignments are equal
        #   (with or without permutation).
        "mutual_info": mutual_info_score(labels_true, labels_pred),  # symmetric
        "normalized_mutual_info": normalized_mutual_info_score(
            labels_true, labels_pred
        ),  # symmetric
        "adjusted_mutual_info": adjusted_mutual_info_score(
            labels_true, labels_pred
        ),  # symmetric
        # https://scikit-learn.org/stable/modules/clustering.html#homogeneity-completeness-and-v-measure
        # * homogeneity: each cluster contains only members of a single class.
        # * completeness: all members of a given class are assigned to the same cluster.
        # * v_measure: harmonic mean of homogeneity and completeness.
        # Bounded scores: 0.0 is as bad as it can be, 1.0 is a perfect score.
        # The V-measure is actually equivalent to the mutual information (NMI) discussed above,
        #   with the aggregation function being the arithmetic mean.
        "completeness": completeness_score(labels_true, labels_pred),
        "homogeneity": homogeneity_score(labels_true, labels_pred),
        "v_measure": v_measure_score(labels_true, labels_pred),  # symmetric
        # https://scikit-learn.org/stable/modules/clustering.html#fowlkes-mallows-scores
        # The Fowlkes-Mallows score FMI is defined as the geometric mean of the pairwise precision and recall.
        #   FMI = TP / sqrt{(TP + FP) * (TP + FN)}
        #   TP: true positive, the number of pair of points that belong to the same clusters in both the true labels and the predicted labels
        #   FP: false positive, the number of pair of points that belong to the same clusters in the true labels and not in the predicted labels
        #   FN: false negative, the number of pair of points that belongs in the same clusters in the predicted labels and not in the true labels
        # The score ranges from 0 to 1. A high value indicates a good similarity between two clusters.
        "fowlkes_mallows_score": fowlkes_mallows_score(labels_true, labels_pred),
    }


def calculate_metrics_without_labels(X, labels_pred):
    """Calculates the clutering measures without knowledge of ground-truth labels.

    Args:
        X (array): A list of n_features-dimensional data points. (n_samples, n_features)
            Each row corresponds to a single data point.
        labels_pred (list): The predicted labels. (n_samples,)
    Returns:
        dict: A dictionary containing the clustering measures.
    """
    return {
        # https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient
        # The Silhouette Coefficient is defined for each sample and is composed of two scores:
        #   s = (b - a) / max(a, b)
        #   a: The mean distance between a sample and all other points in the same class.
        #   b: The mean distance between a sample and all other points in the next nearest cluster.
        # The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering.
        # Scores around zero indicate overlapping clusters.
        # The score is higher when clusters are dense and well separated,
        #   which relates to a standard concept of a cluster.
        "silhouette_score": silhouette_score(X, labels_pred, metric="euclidean"),
        # https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
        # The Calinski-Harabasz score is defined as ratio of the sum of between-clusters dispersion
        #   and of within-cluster dispersion for all clusters.
        #   (where dispersion is defined as the sum of distances squared)
        # The score is higher when clusters are dense and well separated,
        #   which relates to a standard concept of a cluster.
        "calinski_harabasz_score": calinski_harabasz_score(X, labels_pred),
        # https://scikit-learn.org/stable/modules/clustering.html#davies-bouldin-index
        # The Davies-Bouldin index signifies the average 'similarity' between clusters,
        #   where the similarity is a measure that compares the distance between clusters.
        # Zero is the lowest possible score. Values closer to zero indicate a better partition.
        "davies_bouldin_score": davies_bouldin_score(X, labels_pred),
    }


# https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ground_truth_labels",
        type=str,
        help="Path to the text file containing ground truth labels",
    )
    parser.add_argument(
        "clustering_labels",
        type=str,
        help="Path to the text file containing clustering labels",
    )
    args = parser.parse_args()

    with open(args.ground_truth_labels, "r") as f:
        y_true = [int(line.strip()) for line in f if line.strip()]
    with open(args.clustering_labels, "r") as f:
        y_pred = [int(line.strip()) for line in f if line.strip()]
    assert len(y_true) == len(y_pred), (len(y_true), len(y_pred))
    for name, val in calculate_metrics(y_true, y_pred).items():
        print(f"{name}: {val:.3f}")
