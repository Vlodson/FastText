import numpy as np

from embedding import embed_word


def vector_cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Does cosine similarity (from formula of vector product)
    Return interval is -1 to 1 with 1 being same and -1 being completely different
    """
    similarity = vector1 @ vector2.T / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    # due to mat mul the result is a shape(1, 1) np array and not a float, so indexing is needed
    return similarity[0, 0]


def word_similarity(word: str, word_map: Dict[str, Dict[str, np.ndarray]], ngram_vectors: Dict[str, np.ndarray],
                    vector_space: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """
    Finds the cosine similarity between the word and the whole known dictionary in the word_map.
    Returns a dictionary with the key being the word and the value being the similarity between
    the queried word and the word in the key
    """
    embedded_word = embed_word(word, ngram_vectors, vector_space)

    if type(embedded_word) == int:
        raise Exception(f"Word \"{word}\" can't be embedded")

    similarities = {}
    for dict_word in word_map.keys():
        embedded_dict_word = embed_word(dict_word, ngram_vectors, vector_space)
        similarities[dict_word] = vector_cosine_similarity(embedded_word, embedded_dict_word)

    return word, similarities
