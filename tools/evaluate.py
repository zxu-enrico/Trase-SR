"""Compare predicted API sets with the gold sets, and score them.

python tools/evaluate.py --dataset programmableweb               # outputs/programmableweb/predictions_train.jsonl
python tools/evaluate.py --dataset programmableweb --file PATH   # any JSONL with "gold" and "predicted"

Two labels of the same API are one API (tools/mapping.py); a name that is not in the catalog is a wrong recommendation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import data  # noqa: E402
from tools.mapping import Catalog, key  # noqa: E402


def compare(gold, predicted, catalog):
    """gold: api_ids; predicted: [{"name", "api_id"}] from Catalog.parse. -> (wrong, missed): predicted api_ids (names
    when not in the catalog) that are not in the gold set, and gold api_ids that were not predicted."""
    identity = lambda api_id: catalog.identity(catalog.apis[api_id])
    gold_apis = {identity(g) for g in gold}
    predicted_apis = {identity(p["api_id"]) for p in predicted if p["api_id"]}
    wrong = [p["api_id"] or p["name"] for p in predicted if not p["api_id"] or identity(p["api_id"]) not in gold_apis]
    return wrong, [g for g in gold if identity(g) not in predicted_apis]


def scores(rows, catalog):
    """Micro precision, recall and F1, macro F1 and exact-set accuracy over rows with "gold" and "predicted"."""
    hits = n_predicted = n_gold = exact = 0
    macro = 0.0
    for row in rows:
        gold = {catalog.identity(catalog.apis[g]) for g in row["gold"]}
        predicted = {catalog.identity(catalog.apis[p["api_id"]]) if p["api_id"] else "?" + key(p["name"]) for p in row["predicted"]}
        hit = len(gold & predicted)
        hits, n_predicted, n_gold, exact = hits + hit, n_predicted + len(predicted), n_gold + len(gold), exact + (gold == predicted)
        macro += 2 * hit / (len(gold) + len(predicted)) if gold or predicted else 1.0
    precision, recall = hits / max(n_predicted, 1), hits / max(n_gold, 1)
    return {"samples": len(rows), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if hits else 0.0,
            "macro_f1": macro / max(len(rows), 1), "exact": exact / max(len(rows), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=data.DATASETS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--file", type=Path, help="default: outputs/<dataset>/predictions_<split>.jsonl")
    args = parser.parse_args()
    _, apis = data.load(args.dataset, args.split)
    result = scores(data.read_jsonl(args.file or data.output(args.dataset, f"predictions_{args.split}.jsonl")), Catalog(apis))
    print("  ".join(f"{k} {v:.1%}" if isinstance(v, float) else f"{k} {v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
