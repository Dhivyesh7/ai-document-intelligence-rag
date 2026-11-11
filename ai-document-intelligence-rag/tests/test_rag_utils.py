import pytest

from rag_utils import chunk_text


def test_chunk_text_returns_empty_for_blank_input():
    assert chunk_text("   ") == []


def test_chunk_text_splits_and_overlaps():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwxyz"]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", chunk_size=5, overlap=5)
