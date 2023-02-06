import numpy as np

from typing import List, Dict, Tuple, Union

from globals import NGRAM_SIZE, CORPUS_PATH
from text_preprocesssing import preprocess_corpus, word_to_vector
from model import train


def make_ngram_list_from_word(word: str) -> List[str]:
    """
    Returns a list of ngrams as well as the word itself
    """
    ngrams = [word[i:i+NGRAM_SIZE] for i in range(0, len(word) - (NGRAM_SIZE - 1))]
    ngrams.append(word)
    return ngrams


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

    embedded = np.sum(vector_space * word_vectorized.reshape(-1, 1), axis=1) / np.sum(word_vectorized != 0)
    return embedded.reshape(1, -1)


def vector_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Does cosine similarity (from formula of vector product)
    """
    return vector1 @ vector2.T / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def word_similarity(word: str, word_map: Dict[str, Dict[str, np.ndarray]], ngram_vectors: Dict[str, np.ndarray],
                    vector_space: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Finds the cosine similarity between the word and the whole known dictionary in the word_map.
    Returns a dictionary with the key being the word and the value being the similarity between
    the queried word and the word in the key
    """
    embedded_word = embed_word(word, ngram_vectors, vector_space)

    if type(embedded_word) == int:
        raise Exception(f"Word {word} can't be embedded")

    similarities = {}
    for dict_word in word_map.keys():
        embedded_dict_word = embed_word(dict_word, ngram_vectors, vector_space)
        similarities[dict_word] = vector_cosine_similarity(embedded_word, embedded_dict_word)

    return word, similarities


def main():
    ngram_map, context_map, ngram_vectors, word_map = preprocess_corpus(CORPUS_PATH)
    vector_space = train(word_map, True)
    word = [*word_map.keys()][0]
    print(word_similarity("kise", word_map, ngram_vectors, vector_space))


if __name__ == '__main__':
    main()
