"""Apply changed mapping rules (tools/mapping.py) to saved model outputs, without calling a model.

python tools/remap.py --dataset programmableweb [--split train] [--dry-run]

1. outputs/<dataset>/predictions_<split>.jsonl: "predicted" is parsed again from the raw "output".
2. Every error_explanations_<split>.jsonl of the dataset (student and GPT) keeps only the rows whose input is unchanged:
   the sample still has an error, with the same wrong and missed APIs and the same predicted names. Rows stay in
   sample order.
Then rerun generate/explain_errors.py and generate/gpt_explain_errors.py: they generate exactly the missing rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import data  # noqa: E402
from tools.evaluate import compare  # noqa: E402
from tools.mapping import Catalog  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=data.DATASETS)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dry-run", action="store_true", help="report the changes without writing")
    args = parser.parse_args()
    samples, apis = data.load(args.dataset, args.split)
    catalog, order = Catalog(apis), {s["id"]: i for i, s in enumerate(samples)}
    path = data.output(args.dataset, f"predictions_{args.split}.jsonl")
    predictions = data.read_jsonl(path)
    shown_before = {p["id"]: [x["name"] for x in p["predicted"]] for p in predictions}
    changed = 0
    for p in predictions:
        parsed = catalog.parse(p["output"])
        changed += parsed != p["predicted"]
        p["predicted"] = parsed
    print(f"{path.relative_to(data.ROOT)}: {changed} of {len(predictions)} predictions change")
    inputs = {}  # id -> (wrong, missed, predicted names) for the samples that have an error
    for p in predictions:
        wrong, missed = compare(p["gold"], p["predicted"], catalog)
        if wrong or missed:
            inputs[p["id"]] = (wrong, missed, [x["name"] for x in p["predicted"]])
    for file in sorted(data.output(args.dataset).glob(f"**/error_explanations_{args.split}.jsonl")):
        rows = data.read_jsonl(file)
        kept = sorted((r for r in rows if r["id"] in inputs and inputs[r["id"]] == (r["wrong"], r["missed"], shown_before.get(r["id"]))),
                      key=lambda r: order[r["id"]])
        print(f"{file.relative_to(data.ROOT)}: keeps {len(kept)} of {len(rows)} rows; {len(inputs) - len(kept)} to generate")
        if not args.dry_run:
            data.write_jsonl(file, kept)
    if not args.dry_run:
        data.write_jsonl(path, predictions)


if __name__ == "__main__":
    main()
