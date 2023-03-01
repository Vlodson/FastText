import numpy as np

from typing import List, Dict, Union

from globals import NGRAM_SIZE
from text_preprocesssing import word_to_vector


def _iter_len(str_len: int, ngrams: int) -> int:
    return str_len - (ngrams - 1) if str_len > ngrams else 1


def word_list_to_ngram_list(word_list: List[str], ngram_size: int) -> List[str]:
    """
    Given a list of words, turns it into a list of ngrams from each word
    """
    return [
        word[i:i + ngram_size]
        for word in word_list
        for i in range(0, _iter_len(len(word), ngram_size))
    ]


def count_missing_ngrams(word_ngram_list: List[str], ngram_vectors: Dict[str, np.ndarray]) -> int:
    """
    Checks to see how many ngrams from a word are not in the dictionary of known ngrams from the preprocessed corpus
    """
    all_ngrams = ngram_vectors.keys()
    return len(word_ngram_list) - sum([ngram in all_ngrams for ngram in word_ngram_list])


def remove_missing_ngrams(word_ngram_list: List[str], ngram_vectors: Dict[str, np.ndarray]) -> List[str]:
    """
    Removes the ngrams not found in the original ngram dictionary
    """
    all_ngrams = ngram_vectors.keys()
    return [ngram for ngram in word_ngram_list if ngram in all_ngrams]


def word_vector(word: str, ngram_vectors: Dict[str, np.ndarray]) -> Union[np.ndarray, int]:
    """
    Returns the vector representing the word

    If there are more than 2 unknown ngrams then it returns -1.
    2 was chosen because that is the minimum case for a new word that only has one ngram changed compared to its closest
    word in the dictionary
    """
    word_ngram_list = make_ngram_list_from_word(word)
    missing = count_missing_ngrams(word_ngram_list, ngram_vectors)

    if missing > 2:
        return -1

    if missing <= 2:
        word_ngram_list = remove_missing_ngrams(word_ngram_list, ngram_vectors)

    return word_to_vector(word_ngram_list, ngram_vectors)


def embed_word(word: str, ngram_vectors: Dict[str, np.ndarray], vector_space: np.ndarray) -> Union[np.ndarray, int]:
    """
    Returns the embedded representation of the word in the vector space or -1 if the word can't be vectorized
    """
    word_vectorized = word_vector(word, ngram_vectors)

    if type(word_vectorized) == int:
        return -1

    embedded = word_vectorized @ vector_space
    return embedded


def embed_word_map(word_map: Dict[str, Dict[str, np.ndarray]], ngram_vectors: Dict[str, np.ndarray],
                   vector_space: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Embeds the word map made in text preprocessing.

    Returns a dict where the keys are the words and values are their embedded vectors.
    """
    return {word: embed_word(word, ngram_vectors, vector_space) for word in word_map.keys()}
