"""Load source documents into the typed ``Document`` model.

Supported out of the box: Markdown, plain text, and PDF. HTML is optional
(needs the ``html`` extra / trafilatura). Each document gets a stable
``doc_id`` derived from its path relative to the corpus root, so citations and
eval ground-truth can refer to documents by a readable, reproducible id.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import MissingDependencyError, RagEvalError
from ..schemas import Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm"}


def _doc_id_from_path(path: Path, root: Path) -> str:
    """Readable, filesystem-independent id: posix relative path without suffix."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = Path(path.name)
    return rel.with_suffix("").as_posix()


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - base dependency
        raise MissingDependencyError("pdf", "pypdf", "all") from exc

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_html(path: Path) -> str:
    try:
        import trafilatura
    except ImportError as exc:
        raise MissingDependencyError("html", "trafilatura", "html") from exc

    extracted = trafilatura.extract(path.read_text(encoding="utf-8", errors="replace"))
    return extracted or ""


def load_document(path: str | Path, root: str | Path | None = None) -> Document:
    """Load a single file into a ``Document``."""
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown", ".txt"}:
        text = _load_text(path)
    elif suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix in {".html", ".htm"}:
        text = _load_html(path)
    else:
        raise RagEvalError(f"Unsupported file type: {path.suffix} ({path})")

    return Document(
        doc_id=_doc_id_from_path(path, root),
        source_path=str(path),
        text=text.strip(),
        metadata={"suffix": suffix},
    )


def load_corpus(corpus_dir: str | Path) -> list[Document]:
    """Recursively load every supported document under ``corpus_dir``."""
    root = Path(corpus_dir)
    if not root.exists():
        raise RagEvalError(f"Corpus directory not found: {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            doc = load_document(path, root=root)
            if doc.text:  # skip empty / unparseable files
                documents.append(doc)
    return documents
