"""Evaluate B/C explanation and D/E revision on test using dev-selected adapters."""

from __future__ import annotations

import argparse
import gc
import json
import re
from datetime import datetime
from pathlib import Path

import torch

import examples  # adds generate/ and the repository root to sys.path
import common
import predict
from tools import data
from tools.evaluate import scores
from tools.mapping import Catalog, key


API_HEADER = re.compile(r"(?im)^[ \t]*(?:\*\*)?APIs:(?:\*\*)?[ \t]*")
REASON_HEADER = re.compile(r"(?i)^\s*(?:\*\*)?Explanation:(?:\*\*)?\s*")


def split_answer(text):
    headers = list(API_HEADER.finditer(text))
    if not headers:
        return text.strip(), "", False
    header = headers[-1]
    reason = text[:header.start()]
    formatted = bool(REASON_HEADER.match(reason)) and len(headers) == 1
    return REASON_HEADER.sub("", reason, count=1).strip(), text[header.end():].strip(), formatted


def write_json(path, value):
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def generate_rows(path, samples, prompts, catalog, args, adapter=None):
    """Write base-model drafts or auxiliary-task completions in test-set order."""
    explain = adapter is not None
    max_tokens = args.max_new_tokens if explain else predict.MAX_NEW_TOKENS
    print(f"START {path.parent.name}/{path.stem} ({len(samples)} samples)", flush=True)
    tokenizer, net = predict.load_model(args.model, adapter)
    rows = [None] * len(samples)
    order = sorted(range(len(samples)), key=lambda i: len(prompts[i]))
    for offset in range(0, len(order), args.batch_size):
        batch = order[offset:offset + args.batch_size]
        with torch.autocast(predict.DEVICE, dtype=torch.bfloat16):
            outputs = common.greedy(net, tokenizer, [prompts[i] for i in batch], max_tokens)
        for i, (output, usage) in zip(batch, outputs, strict=True):
            sample = samples[i]
            reason, api_text, formatted = split_answer(output) if explain else ("", output, True)
            parsed = catalog.parse(api_text)
            row = {
                "id": sample["id"],
                "gold": [api["api_id"] for api in sample["gold_apis"]],
                "output": output,
                "predicted": parsed,
                **usage,
            }
            if explain:
                row.update(explanation=reason, api_text=api_text, format_ok=bool(formatted and reason and parsed))
            rows[i] = row
        print(f"  {path.parent.name}/{path.stem} {offset + len(batch)}/{len(samples)}", flush=True)
    data.write_jsonl(path, rows)
    del net, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def revision_stats(drafts, revisions, catalog):
    counts = dict(correct_drafts=0, exact_repairs=0, exact_regressions=0,
                  recovered_gold=0, dropped_gold=0, removed_wrong=0, added_wrong=0)

    def identity(api_id):
        return catalog.identity(catalog.apis[api_id])

    def predicted_apis(row):
        return {identity(p["api_id"]) if p["api_id"] else "?" + key(p["name"]) for p in row["predicted"]}

    for draft, revised in zip(drafts, revisions, strict=True):
        if draft["id"] != revised["id"]:
            raise ValueError("Draft and revision IDs differ")
        gold = {identity(api_id) for api_id in draft["gold"]}
        before, after = predicted_apis(draft), predicted_apis(revised)
        counts["correct_drafts"] += int(before == gold)
        counts["exact_repairs"] += int(before != gold and after == gold)
        counts["exact_regressions"] += int(before == gold and after != gold)
        counts["recovered_gold"] += len((gold & after) - before)
        counts["dropped_gold"] += len((gold & before) - after)
        counts["removed_wrong"] += len(before - gold - after)
        counts["added_wrong"] += len(after - gold - before)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=data.DATASETS, help="evaluate one dataset (default: all three)")
    parser.add_argument("--model", default=common.STUDENT, help="base model ID or local model directory")
    parser.add_argument("--batch-size", type=int, default=predict.BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=512, help="limit for an auxiliary-task completion")
    parser.add_argument("--output-dir", type=Path, help="new output directory (default: results/auxiliary_test_<timestamp>)")
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens) < 1:
        parser.error("batch size and max new tokens must be positive")

    run_dir = args.output_dir or predict.RESULTS / datetime.now().strftime("auxiliary_test_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    summary = {
        "split": "test",
        "base_model": args.model,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "draft_max_new_tokens": predict.MAX_NEW_TOKENS,
        "datasets": {},
    }
    datasets = (args.dataset,) if args.dataset else data.DATASETS
    for dataset in datasets:
        samples, apis = data.load(dataset, "test")
        catalog = Catalog(apis)
        out = run_dir / dataset
        kind, api_kind, _ = common.TASKS[dataset]
        drafts = generate_rows(
            out / "drafts.jsonl", samples, [examples.main_prompt(dataset, s["request"]) for s in samples],
            catalog, args,
        )
        entry = {"draft": scores(drafts, catalog)}
        summary["datasets"][dataset] = entry
        write_json(run_dir / "summary.json", summary)
        for condition in "BCDE":
            run = examples.RUNS[condition]
            checkpoint = predict.CHECKPOINTS / dataset / run
            epoch = json.loads((checkpoint / "best.json").read_text(encoding="utf-8"))["best_epoch"]
            if condition in "BC":
                prompts = [
                    examples.EXPLAIN_GOLD.format(request_kind=kind, api_kind=api_kind, request=s["request"])
                    for s in samples
                ]
            else:
                prompts = [
                    examples.EXPLAIN_ERRORS.format(
                        request_kind=kind, request=s["request"],
                        predicted=", ".join(p["name"] for p in d["predicted"]) or "(nothing)",
                    )
                    for s, d in zip(samples, drafts, strict=True)
                ]
            outputs = generate_rows(
                out / f"predictions_{condition}.jsonl", samples, prompts, catalog, args, checkpoint / f"epoch{epoch}",
            )
            metrics = scores(outputs, catalog)
            metrics.update(
                epoch=epoch,
                format_ok=sum(row["format_ok"] for row in outputs),
                has_api_section=sum(bool(API_HEADER.search(row["output"])) for row in outputs),
                truncated=sum(bool(row["truncated"]) for row in outputs),
                mean_completion_tokens=sum(row["completion_tokens"] for row in outputs) / max(len(outputs), 1),
            )
            if condition in "DE":
                metrics.update(revision_stats(drafts, outputs, catalog))
                metrics["draft_f1"] = entry["draft"]["f1"]
                metrics["delta_f1"] = metrics["f1"] - metrics["draft_f1"]
            direct_path = predict.RESULTS / dataset / run / "metrics_test.json"
            if direct_path.is_file():
                direct = json.loads(direct_path.read_text(encoding="utf-8"))
                if direct["samples"] == len(samples) and direct.get("epoch") == epoch:
                    metrics["direct_f1"] = direct["f1"]
            entry[condition] = metrics
            write_json(out / f"metrics_{condition}.json", metrics)
            write_json(run_dir / "summary.json", summary)

    print(f"Results saved to {run_dir}", flush=True)


if __name__ == "__main__":
    main()
