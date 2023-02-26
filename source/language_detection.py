from typing import List, Dict

from .io_utils import serialize_object
from .text_preprocesssing import clean_text, make_word_list
# kada pravis od proizvoljnog korpusa ngram_counts dict, koristi ngrame iz dict jezika sa kojim hoces da poredis
# da bi uopste posle mogao da radis kl divergenciju, jer mozda ce doci sa novim ngramima ili neke nece imati


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


def standardize_distributions(ngram_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Standardizes the ngram counts' values to a 0-1 scale, to be able to do KL divergence later
    """
    total_count = sum(ngram_counts.values())
    return {
        ngram: ngram_counts[ngram] / total_count
        for ngram in ngram_counts
    }


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


def main():
    text = ["pera", "voli", "da", "igra", "kosarku", "sa", "perom"]
    ngrams = word_list_to_ngram_list(text, 3)
    ngrams = count_unique_ngrams(ngrams)
    ngrams = standardize_distributions(ngrams)
    print(ngrams)


if __name__ == "__main__":
    main()
