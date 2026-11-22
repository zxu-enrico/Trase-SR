"""Recommend APIs for a split with a fine-tuned condition (or the base model) and score them.

python train/predict.py --dataset programmableweb --condition D --split test   # the epoch with the best dev F1
python train/predict.py --dataset programmableweb --split dev                  # the base model (zero-shot)

The model sees only the request (the step-1 prompt) and answers with API names: greedy decoding, 128 new tokens.
Writes results/<dataset>/<run>/predictions_<split>.jsonl and metrics_<split>.json (results/<dataset>/zeroshot/ for the
base model); a dev split is tagged with its epoch: predictions_dev_epoch2.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import examples  # noqa: E402  (puts generate/ and the project root on sys.path)
import common  # noqa: E402
from tools import data  # noqa: E402
from tools.evaluate import compare, scores  # noqa: E402
from tools.mapping import Catalog  # noqa: E402

CHECKPOINTS, RESULTS = data.ROOT / "checkpoints", data.ROOT / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS, BATCH_SIZE = 128, 32  # as in step 1 (generate/recommend.py)


def load_model(model, adapter=None):
    """Tokenizer (left padding, for generation) and the bf16 model, with LoRA weights from `adapter` if given."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map=DEVICE)
    if adapter:
        from peft import PeftModel
        net = PeftModel.from_pretrained(net, adapter)
    return tok, net.eval()


def evaluate(net, tok, dataset, split, out, tag, epoch=None, limit=None):
    """Predict one split and score it; writes predictions_<tag>.jsonl and metrics_<tag>.json under `out`."""
    samples, apis = data.load(dataset, split)
    samples, catalog, rows = samples[:limit], Catalog(apis), []
    for i in range(0, len(samples), BATCH_SIZE):
        chunk = samples[i:i + BATCH_SIZE]
        with torch.autocast(DEVICE, dtype=torch.bfloat16):  # as in training: bf16 model, fp32 LoRA weights
            outputs = common.greedy(net, tok, [examples.main_prompt(dataset, s["request"]) for s in chunk], MAX_NEW_TOKENS)
        for s, (text, usage) in zip(chunk, outputs):
            rows.append({"id": s["id"], "gold": [g["api_id"] for g in s["gold_apis"]], "output": text,
                         "predicted": catalog.parse(text), **usage})
    errors = [compare(r["gold"], r["predicted"], catalog) for r in rows]  # (wrong, missed) per sample
    wrong = [x for w, _ in errors for x in w]
    metrics = {**({"epoch": epoch} if epoch else {}), **scores(rows, catalog),
               "wrong_in_catalog": sum(x in apis for x in wrong), "wrong_not_in_catalog": sum(x not in apis for x in wrong),
               "missed": sum(len(m) for _, m in errors), "truncated": sum(r["truncated"] for r in rows)}
    data.write_jsonl(out / f"predictions_{tag}.jsonl", rows)
    (out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=data.DATASETS)
    parser.add_argument("--condition", choices=examples.RUNS, help="omit for the base model")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--epoch", type=int, help="default: the epoch with the best dev F1 (best.json)")
    parser.add_argument("--model", default=common.STUDENT, help="Hugging Face id or local folder of the base model")
    parser.add_argument("--limit", type=int, help="only the first N samples (for a quick test)")
    args = parser.parse_args()
    if args.condition:
        run = examples.RUNS[args.condition]
        checkpoint = CHECKPOINTS / args.dataset / run
        epoch = args.epoch or json.loads((checkpoint / "best.json").read_text(encoding="utf-8"))["best_epoch"]
        tok, net = load_model(args.model, checkpoint / f"epoch{epoch}")
        out, tag = RESULTS / args.dataset / run, args.split if args.split == "test" else f"dev_epoch{epoch}"
    else:
        tok, net = load_model(args.model)
        epoch, out, tag = None, RESULTS / args.dataset / "zeroshot", args.split
    print(json.dumps(evaluate(net, tok, args.dataset, args.split, out, tag, epoch, args.limit), indent=2))


if __name__ == "__main__":
    main()
