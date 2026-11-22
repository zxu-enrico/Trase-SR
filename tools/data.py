"""Dataset and output files: where they are, and reading and writing JSONL."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"programmableweb": "ProgrammableWeb", "apibench_java": "APIBench_Q/java", "apibench_python": "APIBench_Q/python"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    """Replace the file in one step, with LF line ends on every OS so that files compare across machines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8"))
    os.replace(partial, path)


def load(dataset, split):
    """Samples of one split, and the dataset's APIs keyed by api_id."""
    folder = ROOT / "data" / DATASETS[dataset]
    return read_jsonl(folder / f"{split}.jsonl"), {a["api_id"]: a for a in read_jsonl(folder / "apis.jsonl")}


def output(dataset, name=""):
    """outputs/<dataset>/<name>"""
    return ROOT / "outputs" / dataset / name
