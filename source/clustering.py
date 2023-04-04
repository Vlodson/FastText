import numpy as np

from sklearn.cluster import DBSCAN, KMeans
from typing import Dict, Tuple
from globals import ClusteringConfig


def embedded_word_map_dbscan_clustering(embedded_word_map: Dict[str, np.ndarray],
                                        config: ClusteringConfig) -> Tuple[Dict[str, int], DBSCAN]:
    """
    Does dbscan clustering on the embedded word map generated by text preprocessing.
    This method can return -1 for labels, in which case they are labeled as 'noise'

    Returns a dict with words from the word map as keys and their labels as values,
    and the dbscan model fit to the embedded data in a tuple.
    """
    data = np.vstack([*embedded_word_map.values()])
    dbs = DBSCAN(eps=config.data["eps"], min_samples=config.data["min_samples"]).fit(data)
    labels = dbs.labels_
    return {word: label for word, label in zip(embedded_word_map.keys(), labels)}, dbs


def embedded_word_map_kmeans_clustering(embedded_word_map: Dict[str, np.ndarray],
                                        config: ClusteringConfig) -> Dict[str, int]:
    """
    Does KMeans clustering on embedded generated by text preprocessing

    Returns a dict with words from the word map as keys and their labels as values
    """
    data = np.vstack([*embedded_word_map.values()])
    kmeans = KMeans(n_clusters=config.data["clusters"], n_init=config.data["n_times"]).fit(data)
    labels = kmeans.labels_
    return {word: label for word, label in zip(embedded_word_map.keys(), labels)}, kmeans
