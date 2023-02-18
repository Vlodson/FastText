import numpy as np

from sklearn.manifold import TSNE
from typing import Dict


def reduce_dims_tsne(embedded_word_map: Dict[str, np.ndarray], n_dims: int = 3) -> Dict[str, np.ndarray]:
    """
    Reduces the dimensionality of the embedding using tsne
    """
    data = np.vstack([*embedded_word_map.values()])
    reduced = TSNE(n_components=n_dims).fit_transform(data)

    return {
        word: vector
        for word, vector in zip(embedded_word_map.keys(), reduced)
    }
