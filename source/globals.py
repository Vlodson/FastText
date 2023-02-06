import yaml

with open("global_cfg.yaml", 'r') as f:
    GLOBAL_CFG = yaml.safe_load(f)

TEXT_PREPROC = GLOBAL_CFG["text_preprocessing"]
NEURAL_NETWORK = GLOBAL_CFG["neural_network"]

CORPUS_PATH = TEXT_PREPROC["corpus_path"]
STOPWORDS_PATH = TEXT_PREPROC["stopwords_path"]
NGRAM_SIZE = TEXT_PREPROC["ngram_size"]
CONTEXT_WINDOW = TEXT_PREPROC["context_window"]
SELF_CONTEXT = TEXT_PREPROC["self_context"]

VECTOR_SPACE_SIZE = NEURAL_NETWORK["vector_space_size"]
LEARN_RATE = NEURAL_NETWORK["learn_rate"]
EPOCHS = NEURAL_NETWORK["epochs"]
