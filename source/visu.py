import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from typing import Dict

from dimension_reduction import reduce_dims_tsne

pio.renderers.default = "browser"


def plot_embedded_word_map_3d(embedded_word_map: Dict[str, np.ndarray]) -> None:
    """
    Plots embedded word map in 3D space by reducing dimensions with TSNE
    """
    tsne_data = reduce_dims_tsne(embedded_word_map)

    # need to make a pd like dict that can be plotted nicely
    reduced_word_map = {
        i: [data[0], data[1], data[2], word] for i, (word, data) in enumerate(zip(embedded_word_map.keys(), tsne_data))
    }

    df = pd.DataFrame.from_dict(
        reduced_word_map, orient='index', columns=['component_1', 'component_2', 'component_3', 'word']
    )

    fig = px.scatter_3d(df, x='component_1', y='component_2', z='component_3', text='word')
    fig.show()


def plot_clustered_word_map_3d(embedded_word_map: Dict[str, np.ndarray], labeled_word_map: Dict[str, int]) -> None:
    """
    Plots the clusters of the embedded word map in 3D, reduced by TSNE
    """
    tsne_data = reduce_dims_tsne(embedded_word_map)

    reduced_word_map = {
        i: [data[0], data[1], data[2], word, label]
        for i, (word, data, label) in enumerate(zip(embedded_word_map.keys(), tsne_data, labeled_word_map.values()))
    }

    df = pd.DataFrame.from_dict(
        reduced_word_map, orient='index', columns=['component_1', 'component_2', 'component_3', 'word', 'label']
    )

    fig = px.scatter_3d(df, x='component_1', y='component_2', z='component_3', text='word', color='label')
    fig.show()
