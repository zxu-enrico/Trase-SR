"""Step 3 (condition E): the student explains its own wrong and missed APIs from step 1, given the gold set.

python generate/explain_errors.py --dataset programmableweb   ->  outputs/programmableweb/error_explanations_train.jsonl
Reads outputs/programmableweb/predictions_train.jsonl; only samples with at least one error are explained.
"""

import common  # also puts the project root on sys.path, for tools/
from tools.evaluate import compare
from tools.mapping import Catalog

PROMPT = """Here is a {request_kind}:
{request}

A model recommended: {predicted}
The correct {api_kind} are: {gold}

Wrongly recommended:
{wrong}
Missed:
{missed}

For each wrongly recommended API, explain in one or two sentences why it does not fit the request. For each missed API, explain which part of the request needs it. Base the explanations on the documentation."""


def jobs(args):
    """Samples with errors, prompt and record builders; gpt_explain_errors.py uses the same ones."""
    samples, apis = common.load(args.dataset, args.split)
    source = common.output(args.dataset, f"predictions_{args.split}.jsonl")
    if not source.exists():
        raise SystemExit(f"{source} not found: run generate/recommend.py --dataset {args.dataset} first")
    predictions = {p["id"]: p for p in common.read_jsonl(source)}
    request_kind, api_kind, _ = common.TASKS[args.dataset]
    catalog = Catalog(apis)
    # wrong: predicted api_ids (names when not in the dataset) outside the gold set; missed: gold api_ids not predicted
    errors = lambda s: compare([g["api_id"] for g in s["gold_apis"]], predictions[s["id"]]["predicted"], catalog)
    with_errors = [s for s in samples if s["id"] in predictions and any(errors(s))]
    print(f"{len(with_errors)} of {len(predictions)} predicted samples have wrong or missed APIs")
    describe = lambda x: common.doc(apis[x]) if x in apis else f"- {x}: (not an API of this dataset)"

    def prompt(s):
        wrong, missed = errors(s)
        return PROMPT.format(request_kind=request_kind, api_kind=api_kind, request=s["request"],
                             predicted=", ".join(p["name"] for p in predictions[s["id"]]["predicted"]) or "(nothing)",
                             gold=", ".join(g["name"] for g in s["gold_apis"]),
                             wrong="\n".join(map(describe, wrong)) or "(none)", missed="\n".join(map(describe, missed)) or "(none)")

    def record(s, text):
        wrong, missed = errors(s)
        return {"id": s["id"], "gold": [g["api_id"] for g in s["gold_apis"]], "wrong": wrong, "missed": missed, "explanation": text}

    return with_errors, prompt, record


def main():
    args = common.parse_args(__doc__, max_new_tokens=768)
    samples, prompt, record = jobs(args)
    common.run(args, samples, common.output(args.dataset, f"error_explanations_{args.split}.jsonl"), prompt, record)


if __name__ == "__main__":
    main()
