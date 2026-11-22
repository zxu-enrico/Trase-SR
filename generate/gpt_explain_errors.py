"""Condition D: GPT explains the student's wrong and missed APIs from step 1, with exactly the input of explain_errors.py.

python generate/gpt_explain_errors.py --dataset programmableweb   ->  outputs/programmableweb/gpt-6-luna-none/error_explanations_train.jsonl
Reads the student's outputs/programmableweb/predictions_train.jsonl (run recommend.py first).
"""

import common
import explain_errors


def main():
    args = common.parse_args(__doc__, max_new_tokens=768, gpt=True)  # same limit as explain_errors.py
    samples, prompt, record = explain_errors.jobs(args)
    path = common.output(args.dataset, f"{args.model}-{args.reasoning_effort}/error_explanations_{args.split}.jsonl")
    common.run(args, samples, path, prompt, record, backend=common.GPT)


if __name__ == "__main__":
    main()
