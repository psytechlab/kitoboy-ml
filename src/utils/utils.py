"""This module defines various util functions."""
import json
from pathlib import Path
from typing import Any, Dict, Union


def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read data from json file.

    Args:
        file_path: File path to read from.
    Returns:
        Dictionary read from json.
    """
    data = None
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)

    return data


def save_to_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Write data to json file.

    Args:
     file_path: File path to write to.
     data: Data to write.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def critical_punctuation_norm(text: str) -> str: 
    return text.replace("…", "...").replace("‚", ",").replace("‸", ".")


def save_text(text: str, path: str):
    """Save text into file.

    Args:
        text (str): The text to be saved.
        path (str): The path to file
    """
    with open(path, "w") as f:
        f.write(text)