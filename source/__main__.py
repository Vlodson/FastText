from .text_node_with_keyword import node_has_keywords


def main():
    with open("../texts/NM_test-corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    print(node_has_keywords(text, ["lorem", "pera", "consectetum"]))


if __name__ == "__main__":
    main()
