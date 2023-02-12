from model import train
from text_preprocesssing import preprocess_corpus
from text_node_with_keyword import make_node_word_list, keywords_similarity, keyword_in_node


def main():
    ngram_map, context_map, ngram_vectors, word_vectors, word_map = preprocess_corpus(
        "../texts/NM_train-corpus-short.txt"
    )

    vector_space = train(word_map, False)
    node_words = make_node_word_list("../texts/NM_test-corpus.txt")

    keyword_similarities = keywords_similarity(
        ["lorem", "pera", "consectetum"], node_words, ngram_vectors, vector_space
    )

    print(keyword_in_node(keyword_similarities, 0.1))


if __name__ == "__main__":
    main()
