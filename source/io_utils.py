import os
import pickle

from typing import Any

from globals import ROOT, SAVE_DIR


def __make_dir(dir_path: str) -> None:
    """
    Checks if path dir exists, and if not, makes it
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def __check_dirs() -> None:
    """
    Makes all the needed directories for serialization to work
    """
    __make_dir(os.path.join(ROOT, SAVE_DIR))


def serialize_object(obj: Any, path: str) -> None:
    """
    Serializes object to pickle file
    """
    __check_dirs()
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def deserialize_object(path: str) -> Any:
    """
    Deserializes from a pickle file to any object.
    Assumes path is exists.
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    return obj


def open_corpus(path: str) -> str:
    """
    Opens a corpus and loads its content into a string.
    Assumes path is exists.
    """
    with open(path, 'r') as f:
        content = f.read()

    return content
