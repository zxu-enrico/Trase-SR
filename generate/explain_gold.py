"""Step 2 (condition C): the student explains why the correct APIs fit the request; its own predictions are not shown.

python generate/explain_gold.py --dataset programmableweb   ->  outputs/programmableweb/gold_explanations_train.jsonl
"""

import common

PROMPT = """Here is a {request_kind}:
{request}

The correct {api_kind} for it are:
{apis}

For each API, explain in one or two sentences which part of the request it serves, based on its documentation."""


def jobs(args):
    """Samples, prompt and record builders; gpt_explain_gold.py uses the same ones so both see identical input."""
    samples, apis = common.load(args.dataset, args.split)
    request_kind, api_kind, _ = common.TASKS[args.dataset]
    prompt = lambda s: PROMPT.format(request_kind=request_kind, api_kind=api_kind, request=s["request"],
                                     apis="\n".join(common.doc(apis[g["api_id"]]) for g in s["gold_apis"]))
    record = lambda s, text: {"id": s["id"], "gold": [g["api_id"] for g in s["gold_apis"]], "explanation": text}
    return samples, prompt, record


def main():
    args = common.parse_args(__doc__, max_new_tokens=768)
    samples, prompt, record = jobs(args)
    common.run(args, samples, common.output(args.dataset, f"gold_explanations_{args.split}.jsonl"), prompt, record)


if __name__ == "__main__":
    main()
