"""Condition B: GPT explains why the correct APIs fit the request, with exactly the input of explain_gold.py.

python generate/gpt_explain_gold.py --dataset programmableweb   ->  outputs/programmableweb/gpt-6-luna-none/gold_explanations_train.jsonl
(the folder is named after --model and --reasoning-effort)
"""

import common
import explain_gold


def main():
    args = common.parse_args(__doc__, max_new_tokens=768, gpt=True)  # same limit as explain_gold.py
    samples, prompt, record = explain_gold.jobs(args)
    path = common.output(args.dataset, f"{args.model}-{args.reasoning_effort}/gold_explanations_{args.split}.jsonl")
    common.run(args, samples, path, prompt, record, backend=common.GPT)


if __name__ == "__main__":
    main()
