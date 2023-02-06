import numpy as np
import re
import string

from typing import List, Dict, Tuple
from operator import itemgetter

from globals import CORPUS_PATH, STOPWORDS_PATH, NGRAM_SIZE, CONTEXT_WINDOW, SELF_CONTEXT


def load_corpus(corpus_path: str) -> str:
    """
    Loads corpus text from text file with path corpus_path 
    """
    with open(corpus_path, 'r') as f:
        corpus = f.read()

    return corpus


def remove_white_space(text: str) -> str:
    """
    Removes multiple occurrences of space, newlines, return lines with a single space
    """
    cleaned_text = re.sub(r"(\s+|\n+|\r+)", ' ', text)
    return cleaned_text


def remove_numbers(text: str) -> str:
    """
    Removes all numbers in text
    """
    cleaned_text = re.sub(r"[1-9]+", ' ', text)
    return cleaned_text


def remove_punctuation(text: str) -> str:
    """
    Removes all occurrences of punctuation
    """
    cleaned_text = text.translate(str.maketrans('', '', string.punctuation))
    return cleaned_text


def to_lowercase(text: str) -> str:
    """
    Turns all words into lowercase
    """
    return text.lower()


def clean_text(text: str) -> str:
    """
    Cleans text using above functions
    """
    text = remove_white_space(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    return text


def make_word_list(text: str) -> List[str]:
    """
    Makes a list from single whitespace separated text
    """
    split_text = text.split(' ')
    return [word for word in split_text if word != '']


def remove_stopwords(word_list: List[str], stopwords_path: str) -> List[str]:
    """
    Removes all stopwords from text.
    Stopwords path is the path to the comma separated stopwords.
    """
    # preprocess stopwords
    with open(stopwords_path, 'r') as f:
        stopwords = f.read()
    stopwords = stopwords.split(',')

    word_list = [word for word in word_list if word not in stopwords]
    return word_list


def make_ngram_map(word_list: List[str], ngram_size: int) -> Dict[str, List[str]]:
    """
    Makes a map of all the words and their ngrams given a list of words.
    Inside the list of ngrams is also the word itself as a ngram (as per facebook docs)
    """
    ngram_map = {}
    for word in word_list:
        ngram_map[word] = [word[i:i+ngram_size] for i in range(0, len(word) - (ngram_size - 1))]
        ngram_map[word].append(word)
        word_list = [other_word for other_word in word_list if other_word != word]

    return ngram_map


def make_context_map(word_list: List[str], left_context_window: int, right_context_window: int) -> Dict[str, List[str]]:
    """
    Makes a map of the context of a single word given how far left and right the context stretches.
    The word itself is inside its own context.
    """
    context_map = {}

    for i, word in enumerate(word_list):
        temp_left_context_window = left_context_window if i > left_context_window else i

        if word not in context_map.keys():
            context_map[word] = word_list[i - temp_left_context_window:i + right_context_window + 1]
        else:
            context_map[word].extend(word_list[i - temp_left_context_window:i + right_context_window + 1])

        if not SELF_CONTEXT:
            context_map[word].remove(word)

        # removes duplicates, but could be slow
        context_map[word] = list(set(context_map[word]))

    return context_map


def make_ngram_vector_map(ngram_map: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    """
    Makes a vector for each ngram and maps the ngram to that vector.
    """
    # python magic. list of lists -> list of all elements using iterators
    all_ngrams = [ngram for ngram_list in [*ngram_map.values()] for ngram in ngram_list]
    unique_ngrams = np.unique(all_ngrams)
    unique_ngram_number = unique_ngrams.shape[0]

    ngram_vector_map = {}
    for i, ngram in enumerate(unique_ngrams):
        # dtype float because numpy matrix operations later will all be in floats
        ngram_vector = np.zeros(shape=(1, unique_ngram_number), dtype=float)
        ngram_vector[0, i] = 1.
        ngram_vector_map[ngram] = ngram_vector

    return ngram_vector_map


def word_to_vector(ngram_list: List[str], ngram_vector_map: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Given a list of ngrams of a word, returns the vector representing that word based on ngrams
    """
    # more python magic. itemgetter gets a collection of items from an iterable/collection all using iterables
    return np.sum(itemgetter(*ngram_list)(ngram_vector_map), axis=0).reshape(1, -1)


def context_to_target(context_vectors: List[np.ndarray]) -> np.ndarray:
    """
    Given context words returns an array representing the target(s) for the input word
    """
    return np.vstack(context_vectors)


def make_word_map(word_list: List[str], ngram_map: Dict[str, List[str]], context_map: Dict[str, List[str]],
                  ngram_vectors: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Maps the input vector as well as the target matrix to their respective words

    Return dict in the shape of: {word: {input: vector, context: matrix}}
    """
    word_map = {}
    for word in np.unique(word_list):
        single_entry = {}
        ngrams = ngram_map[word]
        context = context_map[word]

        input_vector = word_to_vector(ngrams, ngram_vectors)
        target_matrix = context_to_target(
            [word_to_vector(ngram_map[context_word], ngram_vectors) for context_word in context]
        )

        single_entry["input"] = input_vector
        single_entry["context"] = target_matrix

        word_map[word] = single_entry

    return word_map


def make_maps(word_list: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, np.ndarray],
                                             Dict[str, Dict[str, np.ndarray]]]:
    """
    Makes all the needed maps for later use in the network.

    Returns ngram, context, ngram_vector and word maps
    """
    ngram_map = make_ngram_map(word_list, NGRAM_SIZE)
    context_map = make_context_map(word_list, left_context_window=CONTEXT_WINDOW["left"],
                                   right_context_window=CONTEXT_WINDOW["right"])
    ngram_vectors = make_ngram_vector_map(ngram_map)
    word_map = make_word_map(word_list, ngram_map, context_map, ngram_vectors)

    return ngram_map, context_map, ngram_vectors, word_map


def preprocess_corpus(corpus_path: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, np.ndarray],
                                                 Dict[str, Dict[str, np.ndarray]]]:
    """
    Given the path to the corpus, create dataset that will later be used in the NN
    """
    text = load_corpus(corpus_path)
    text = clean_text(text)
    word_list = make_word_list(text)
    word_list = remove_stopwords(word_list, STOPWORDS_PATH)
    maps = make_maps(word_list)

    return maps


def main():
    maps = preprocess_corpus(CORPUS_PATH)


if __name__ == '__main__':
    main()
