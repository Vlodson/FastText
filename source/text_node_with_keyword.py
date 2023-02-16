import os
from typing import List, Dict, Tuple

import numpy as np

from .text_preprocesssing import clean_text, make_word_list, remove_stopwords
from .embedding import embed_word
from .similarity import vector_cosine_similarity
from .io_utils import deserialize_object
from .globals import (
    STOPWORDS_PATH,
    ROOT,
    SAVE_DIR,
    VECTOR_SPACE_FILE,
    NGRAM_VECTORS_FILE,
)


def _make_node_word_list(text: str) -> List[str]:
    """
    Makes a list of words from the text node that needs to be processed
    """
    text = clean_text(text)
    word_list = make_word_list(text)
    word_list = remove_stopwords(word_list, STOPWORDS_PATH)

    return word_list


def _keywords_similarity(
    keywords: List[str],
    node_word_list: List[str],
    ngram_vectors: Dict[str, np.ndarray],
    vector_space: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Embeds each word in the node word list and each keyword.
    Then for each embedded keyword finds the similarity between each of the words in the node word list.

    Returns a dict with keys being keywords and values being dicts with node words as keys and their cosine
    similarity as values.
    """
    embedded_keywords = {
        keyword: embedding
        for keyword in keywords
        if not isinstance(
            embedding := embed_word(keyword, ngram_vectors, vector_space), int
        )
    }

    embedded_node_words = {
        word: embedding
        for word in node_word_list
        if not isinstance(
            embedding := embed_word(word, ngram_vectors, vector_space), int
        )
    }

    keyword_similarities = {
        keyword: {
            word: vector_cosine_similarity(embedded_keyword, embedded_word)
            for word, embedded_word in embedded_node_words.items()
        }
        for keyword, embedded_keyword in embedded_keywords.items()
    }

    return keyword_similarities


def _keyword_in_node(
    keyword_similarities: Dict[str, Dict[str, float]], threshold: float = 0.1
) -> List[Tuple[str, bool]]:
    """
    Based on the keyword similarities dict and a similarity threshold,
    returns a list of tuples with the keyword and whether the node has that keyword in it
    """
    # if the similarity is over 0.1 for at least one word in a keyword similarity dict then mark that one as True
    keyword_nodes = [
        (
            keyword,
            sum(
                [value >= threshold for value in keyword_similarities[keyword].values()]
            )
            >= 1,
        )
        for keyword in keyword_similarities
    ]
    return keyword_nodes


def node_has_keywords(node_text: str, keywords: List[str]) -> List[Tuple[str, bool]]:
    """
    Based on node content and users keywords, marks whether the node has any keywords.

    Returns a list of tuples with the keyword and whether the node has that keyword in it
    """
    project_path = os.path.join(ROOT, SAVE_DIR)
    ngram_vectors: Dict[str, np.ndarray] = deserialize_object(
        os.path.join(project_path, NGRAM_VECTORS_FILE)
    )
    vector_space: np.ndarray = deserialize_object(
        os.path.join(project_path, VECTOR_SPACE_FILE)
    )

    node_word_list = _make_node_word_list(node_text)
    keyword_similarities = _keywords_similarity(
        keywords, node_word_list, ngram_vectors, vector_space
    )
    keyword_in_node = _keyword_in_node(keyword_similarities)

    return keyword_in_node
