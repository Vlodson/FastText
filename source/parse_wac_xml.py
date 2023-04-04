from bs4 import Tag
from bs4 import BeautifulSoup
import re

from globals import LanguageConfig


def parse_body(body: Tag) -> str:
    text = body.contents[0]
    text = re.sub(r"(\s+|\n+|\t+|\r+)", ' ', text)

    words = text.split(' ')[1:-1]
    first_col_words = [word for word in words[::4]]
    remade = ""
    for word in first_col_words:
        remade += word + ' '

    return remade


def parse_xml(xml_path: str) -> str:
    with open(xml_path, 'r', encoding='utf-8') as f:
        data = f.read()

    xml_data = BeautifulSoup(data, 'xml')
    bodies = xml_data.find_all('s')

    parsed_bodies = [parse_body(body) for body in bodies]

    corpus = ""
    for body in parsed_bodies:
        corpus += body + '\n'

    return corpus


def main():
    corpus = parse_xml("../../WaC/srWaC1.1.06_short.xml")
    cfg = LanguageConfig(language="serbian")
    with open(cfg.corpus_path, 'w', encoding='utf-8') as f:
        f.write(corpus)


if __name__ == '__main__':
    main()
