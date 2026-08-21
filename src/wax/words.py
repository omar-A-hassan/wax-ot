"""English word list used for characterizing U-WaX subspaces in terms of text.

The list is the 'google-10000-english-usa' frequency list shipped inside the package
as ``wax/data/english_words.txt``.
"""

from __future__ import annotations

from .datasets import pkg_data_path

__all__ = ["load_words", "words_path"]


def words_path() -> str:
    """Absolute path of the bundled word list."""
    return pkg_data_path("english_words.txt")


def load_words(path: str | None = None) -> list[str]:
    """Return the bundled frequency word list, lowercased and de-duplicated."""
    if path is None:
        path = words_path()
    words: list[str] = []
    seen: set[str] = set()
    with open(path) as f:
        for line in f:
            w = line.strip().lower()
            if w and w not in seen:
                seen.add(w)
                words.append(w)
    return words