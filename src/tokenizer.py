"""Token counting using tiktoken."""

import tiktoken


_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def get_encoding(name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get a cached tiktoken encoding."""
    if name not in _ENCODING_CACHE:
        _ENCODING_CACHE[name] = tiktoken.get_encoding(name)
    return _ENCODING_CACHE[name]


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in a text string."""
    enc = get_encoding(encoding)
    return len(enc.encode(text))
