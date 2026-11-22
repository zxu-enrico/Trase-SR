"""Step 1: the student recommends APIs for each request on its own (no gold set, no candidate list).

python generate/recommend.py --dataset programmableweb   ->  outputs/programmableweb/predictions_train.jsonl
"""

import common  # also puts the project root on sys.path, for tools/
from tools.mapping import Catalog

PROMPT = """Here is a {request_kind}:
{request}

Which {api_kind} should be used to implement it? Answer with the API names only, one per line, written like "{example}"."""


def main():
    args = common.parse_args(__doc__, max_new_tokens=128)
    samples, apis = common.load(args.dataset, args.split)
    catalog = Catalog(apis)
    request_kind, api_kind, example = common.TASKS[args.dataset]
    common.run(args, samples, common.output(args.dataset, f"predictions_{args.split}.jsonl"),
               prompt=lambda s: PROMPT.format(request_kind=request_kind, api_kind=api_kind, example=example, request=s["request"]),
               record=lambda s, text: {"id": s["id"], "gold": [g["api_id"] for g in s["gold_apis"]],
                                       "output": text, "predicted": catalog.parse(text)})


if __name__ == "__main__":
    main()
