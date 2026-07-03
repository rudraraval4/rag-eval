from rag_eval.config import ChunkingConfig
from rag_eval.ingest.chunker import chunk_document
from rag_eval.schemas import Document


def _doc(text: str) -> Document:
    return Document(doc_id="d", source_path="d.txt", text=text)


def test_chunks_cover_document_and_have_valid_spans():
    text = " ".join(f"word{i}" for i in range(100))
    doc = _doc(text)
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=20, chunk_overlap=5))

    assert len(chunks) > 1
    # Spans must slice back to the exact chunk text.
    for c in chunks:
        assert doc.text[c.start_char : c.end_char] == c.text
        assert c.start_char < c.end_char
    # First chunk starts at 0; chunk indices are contiguous.
    assert chunks[0].start_char == 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Last chunk reaches the end of the document.
    assert chunks[-1].end_char == len(text)


def test_overlap_shares_tokens_between_adjacent_chunks():
    text = " ".join(f"w{i}" for i in range(60))
    chunks = chunk_document(_doc(text), ChunkingConfig(chunk_size=20, chunk_overlap=10))
    # With overlap, the second chunk should start before the first chunk ends.
    assert chunks[1].start_char < chunks[0].end_char


def test_no_overlap_produces_disjoint_chunks():
    text = " ".join(f"w{i}" for i in range(40))
    chunks = chunk_document(_doc(text), ChunkingConfig(chunk_size=10, chunk_overlap=0))
    assert chunks[1].start_char >= chunks[0].end_char


def test_short_document_is_one_chunk():
    chunks = chunk_document(_doc("just a few words"), ChunkingConfig(chunk_size=512))
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words"


def test_empty_document_yields_no_chunks():
    assert chunk_document(_doc("   "), ChunkingConfig()) == []


def test_invalid_overlap_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(_doc("a b c"), ChunkingConfig(chunk_size=10, chunk_overlap=10))
