import numpy as np

from typing import List, Dict, Tuple

from text_preprocesssing import load_corpus, clean_text, make_word_list, remove_stopwords
from similarity import embed_word, vector_cosine_similarity
from globals import STOPWORDS_PATH


def make_node_word_list(node_path: str) -> List[str]:
    """
    Makes a list of words from the text node that needs to be processed
    """
    text = load_corpus(node_path)
    text = clean_text(text)
    word_list = make_word_list(text)
    word_list = remove_stopwords(word_list, STOPWORDS_PATH)

    return word_list


def keywords_similarity(keywords: List[str], node_word_list: List[str], ngram_vectors: Dict[str, np.ndarray],
                        vector_space: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Embeds each word in the node word list and each keyword.
    Then for each embedded keyword finds the similarity between each of the words in the node word list.

    Returns a dict with keys being keywords and values being dicts with node words as keys and their cosine
    similarity as values.
    """
    # I don't think there's a way to not do this twice and still keep it as a generator, luckily it's a quick operation
    # this removes all the ones that couldn't be embedded in this context
    embedded_keywords = {
        keyword: embed_word(keyword, ngram_vectors, vector_space)
        for keyword in keywords if type(embed_word(keyword, ngram_vectors, vector_space)) != int
    }

    embedded_node_words = {
        word: embed_word(word, ngram_vectors, vector_space)
        for word in node_word_list if type(embed_word(word, ngram_vectors, vector_space)) != int
    }

    keyword_similarities = {
        keyword:
            {
                word: vector_cosine_similarity(embedded_keyword, embedded_word)
                for word, embedded_word in embedded_node_words.items()
            }
        for keyword, embedded_keyword in embedded_keywords.items()
    }

    return keyword_similarities


def keyword_in_node(keyword_similarities: Dict[str, Dict[str, float]],
                    threshold: float = 0.1) -> List[Tuple[str, bool]]:
    """
    Based on the keyword similarities dict and a similarity threshold,
    returns a list of tuples with the keyword and whether the node has that keyword in it
    """
    # if the similarity is over 0.1 for at least one word in a keyword similarity dict then mark that one as True
    keyword_nodes = [
        (keyword, sum([value >= threshold for value in keyword_similarities[keyword].values()]) >= 1)
        for keyword in keyword_similarities
    ]
    return keyword_nodes
