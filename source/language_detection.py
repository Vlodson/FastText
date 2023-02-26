from typing import List, Dict

import numpy as np

from .io_utils import serialize_object, deserialize_object
from .text_preprocesssing import clean_text, make_word_list


def word_list_to_ngram_list(word_list: List[str], ngram_size: int) -> List[str]:
    """
    Given a list of words, turns it into a list of ngrams from each word
    """
    iter_len = lambda str_len, ngrams: str_len - (ngrams - 1) if str_len > ngram_size else 1

    return [
        word[i : i + ngram_size]
        for word in word_list
        for i in range(0, iter_len(len(word), ngram_size))
    ]


def count_unique_ngrams(ngram_list: List[str]) -> Dict[str, int]:
    """
    Counts up all the occurrences of unique ngrams
    """
    return {
        ngram: ngram_list.count(ngram)
        for ngram in ngram_list
    }


def count_unique_ngrams_for_unknown_lang(unknown_ngram_list: List[str], known_ngram_list: List[str]) -> Dict[str, int]:
    """
    Counts up all the occurrences of unique ngrams that correspond with ngrams known in the training set of a language.
    This is done so that KL divergence can later be done on both dicts,
    because the unknown language might have new ngrams or not have known ngrams in it
    """
    return {
        ngram: unknown_ngram_list.count(ngram)
        for ngram in known_ngram_list
    }


def standardize_distributions(ngram_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Standardizes the ngram counts' values to a 0-1 scale, to be able to do KL divergence later
    """
    total_count = sum(ngram_counts.values())
    return {
        ngram: ngram_counts[ngram] / total_count
        for ngram in ngram_counts
    }


def kl_divergence(dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
    """
    Computes KL divergence given two standardized ngram counts.
    Note1: KL is always non-negative and that the best score is 1.0
    Note2: to avoid division with 0 and to avoid log(0) a punishment of 100 is added to those with value 0
    """
    return sum([
        dist1[key1] * np.log(dist1[key1]) / np.log(dist2[key2]) if dist1[key1] != 0 else 1e2
        for key1, key2 in zip(dist1, dist2)
    ])


def make_language_ngram_distribution(language_corpus: str, ngram_size: int, distribution_path: str) -> None:
    """
    Given a corpus that represents the corpus that the network is trained on, clean it,
    and find the distribution of ngrams in it then save the distribution
    """
    cleaned = clean_text(language_corpus)
    word_list = make_word_list(cleaned)
    ngram_list = word_list_to_ngram_list(word_list, ngram_size)
    ngram_count = count_unique_ngrams(ngram_list)
    ngram_std = standardize_distributions(ngram_count)
    serialize_object(ngram_std, distribution_path)


def calculate_language_confidence(unknown_corpus: str, ngram_size: int, language_distribution_path: str) -> float:
    """
    Given a corpus of unknown language, calculates the kl divergence w.r.t a known language's distribution
    """
    known_language_dist: Dict[str, float] = deserialize_object(language_distribution_path)

    cleaned = clean_text(unknown_corpus)
    word_list = make_word_list(cleaned)
    ngram_list = word_list_to_ngram_list(word_list, ngram_size)
    ngram_count = count_unique_ngrams_for_unknown_lang(ngram_list, known_language_dist)
    unknown_language_dist = standardize_distributions(ngram_count)

    return kl_divergence(unknown_language_dist, known_language_dist)


def main():
    known = ["pera", "voli", "da", "igra", "kosarku", "sa", "perom"]
    unknown = ["pera", "voli", "da", "igra", "kosarku", "sa", "perom"]

    kn_ngrams = word_list_to_ngram_list(known, 3)
    kn_ngrams = count_unique_ngrams(kn_ngrams)
    kn_std = standardize_distributions(kn_ngrams)

    uk_ngrams = word_list_to_ngram_list(unknown, 3)
    uk_ngrams = count_unique_ngrams_for_unknown_lang(uk_ngrams, kn_ngrams)
    uk_std = standardize_distributions(uk_ngrams)

    err = kl_divergence(uk_std, kn_std)

    print(err)


if __name__ == "__main__":
    main()
