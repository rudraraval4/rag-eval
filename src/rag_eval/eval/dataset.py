"""Load the labeled evaluation set from JSONL.

Each line is one ``EvalCase``: a question, its ideal answer, and the
ground-truth ``relevant_doc_ids`` a good retriever should surface. Keeping the
eval set committed alongside the corpus is what makes results reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import RagEvalError
from ..schemas import EvalCase


def load_eval_set(path: str | Path) -> list[EvalCase]:
    p = Path(path)
    if not p.exists():
        raise RagEvalError(f"Eval set not found: {p}")

    cases: list[EvalCase] = []
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            cases.append(EvalCase.model_validate_json(line))
        except Exception as exc:  # noqa: BLE001 - surface which line is bad
            raise RagEvalError(f"Invalid eval case on line {line_no}: {exc}") from exc
    if not cases:
        raise RagEvalError(f"Eval set is empty: {p}")
    return cases
