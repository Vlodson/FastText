import os

import yaml

globals_yaml_path = os.getenv("GLOBALS_YAML_PATH", "source/global_cfg.yaml")

with open(globals_yaml_path, "rb") as f:
    GLOBAL_CFG = yaml.safe_load(f)

TEXT_PREPROC = GLOBAL_CFG["text_preprocessing"]
NEURAL_NETWORK = GLOBAL_CFG["neural_network"]
IO = GLOBAL_CFG["io"]
CLUSTERING = GLOBAL_CFG["clustering"]

CORPUS_PATH = TEXT_PREPROC["corpus_path"]
STOPWORDS_PATH = TEXT_PREPROC["stopwords_path"]
NGRAM_SIZE = TEXT_PREPROC["ngram_size"]
CONTEXT_WINDOW = TEXT_PREPROC["context_window"]
SELF_CONTEXT = TEXT_PREPROC["self_context"]

VECTOR_SPACE_SIZE = NEURAL_NETWORK["vector_space_size"]
OPTIMIZER = NEURAL_NETWORK["optimizer"]
LEARN_RATE = NEURAL_NETWORK["learn_rate"]
EPOCHS = NEURAL_NETWORK["epochs"]

ROOT = IO["root"]
SAVE_DIR = IO["save_dir"]
VECTOR_SPACE_FILE = IO["vector_space_file"]
NGRAM_VECTORS_FILE = IO["ngram_vectors_file"]
WORD_MAP_FILE = IO["word_map_file"]

DB_SCAN = CLUSTERING["dbscan"]
KMEANS = CLUSTERING["kmeans"]
