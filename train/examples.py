"""Training examples for the five conditions of TRACE_SR.pdf.

Every condition has the main task for every training sample: request -> correct API set, with the step-1 prompt of
generate/recommend.py (the only prompt used at inference). B-E add one explanation example for each sample whose frozen
student draft has an error (the same samples in all four, so their training data are matched):
  B / C  request              -> correct-answer explanation (GPT / student) + correct API set
  D / E  request + the draft  -> error explanation (GPT / student) + correct API set
Explanations are cut at the last whole line within a token budget, the same for B-E.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "generate"))
import common  # noqa: E402  (also puts the project root on sys.path, for tools/)
import recommend  # noqa: E402

RUNS = {"A": "A-labels", "B": "B-gpt-gold", "C": "C-student-gold", "D": "D-gpt-error", "E": "E-student-error"}
GPT = "gpt-6-luna-none"  # folder of the GPT explanations in outputs/<dataset>/
EXPLANATIONS = {"B": f"{GPT}/gold_explanations_train.jsonl", "C": "gold_explanations_train.jsonl",
                "D": f"{GPT}/error_explanations_train.jsonl", "E": "error_explanations_train.jsonl"}

EXPLAIN_GOLD = """Here is a {request_kind}:
{request}

Which {api_kind} should be used to implement it? First explain which part of the request each API serves, then list the API names, one per line, after "APIs:"."""

EXPLAIN_ERRORS = """Here is a {request_kind}:
{request}

A model recommended: {predicted}

Check this recommendation: explain which recommended APIs do not fit the request and which needed APIs are missing, then list the correct API names, one per line, after "APIs:"."""


def main_prompt(dataset, request):
    """The step-1 prompt: request -> API names. Training, dev and test all use it."""
    request_kind, api_kind, example = common.TASKS[dataset]
    return recommend.PROMPT.format(request_kind=request_kind, api_kind=api_kind, example=example, request=request)


def answer(sample):
    return "\n".join(g["name"] for g in sample["gold_apis"])


def cut(text, tok, limit):
    """The text up to its last whole line within `limit` tokens -> (text, whether it was cut)."""
    text = text.strip()
    if len(tok(text, add_special_tokens=False).input_ids) <= limit:
        return text, False
    kept = []
    for line in text.splitlines():
        if len(tok("\n".join(kept + [line]), add_special_tokens=False).input_ids) > limit:
            break
        kept.append(line)
    if not "".join(kept).strip():  # the first line alone is too long: cut inside it
        kept = [tok.decode(tok(text, add_special_tokens=False).input_ids[:limit])]
    return "\n".join(kept).strip(), True


def build(dataset, condition, tok, explanation_tokens=384, limit=None):
    """[{"id", "task", "prompt", "target"}] for one condition, and the counts for the run's config.
    limit: only the first N training samples (for a quick test)."""
    samples = common.load(dataset, "train")[0][:limit]
    examples = [{"id": s["id"], "task": "main", "prompt": main_prompt(dataset, s["request"]), "target": answer(s)} for s in samples]
    counts = {"main": len(examples), "explain": 0, "explanations_cut": 0}
    if condition == "A":
        return examples, counts
    # the samples whose draft has an error: those explained in D and E
    with_errors = [{r["id"] for r in common.read_jsonl(common.output(dataset, EXPLANATIONS[c]))} for c in "DE"]
    if with_errors[0] != with_errors[1]:
        raise SystemExit(f"D and E explain different samples in outputs/{dataset}/; regenerate them first")
    texts = {r["id"]: r["explanation"] for r in common.read_jsonl(common.output(dataset, EXPLANATIONS[condition]))}
    drafts = {r["id"]: r["predicted"] for r in common.read_jsonl(common.output(dataset, "predictions_train.jsonl"))}
    request_kind, api_kind, _ = common.TASKS[dataset]
    for s in samples:
        if s["id"] not in with_errors[0]:
            continue
        text, was_cut = cut(texts[s["id"]], tok, explanation_tokens)
        if condition in "BC":
            prompt = EXPLAIN_GOLD.format(request_kind=request_kind, api_kind=api_kind, request=s["request"])
        else:  # the draft as the explainer saw it (generate/explain_errors.py)
            predicted = ", ".join(p["name"] for p in drafts[s["id"]]) or "(nothing)"
            prompt = EXPLAIN_ERRORS.format(request_kind=request_kind, request=s["request"], predicted=predicted)
        examples.append({"id": s["id"], "task": "explain", "prompt": prompt, "target": f"Explanation:\n{text}\nAPIs:\n{answer(s)}"})
        counts["explain"] += 1
        counts["explanations_cut"] += was_cut
    return examples, counts
