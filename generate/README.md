# generate

The student model (Llama-3.2-3B-Instruct) recommends APIs; then the student and GPT each explain the correct APIs and the student's mistakes.

| Script | Condition | Who explains | What |
| --- | --- | --- | --- |
| `recommend.py` | (step 1) | student | recommends APIs from the request alone |
| `explain_gold.py` | C | student | why the correct APIs fit the request |
| `explain_errors.py` | E | student | why its step-1 APIs were wrong or missed |
| `gpt_explain_gold.py` | B | GPT | same input as `explain_gold.py` |
| `gpt_explain_errors.py` | D | GPT | same input as `explain_errors.py` (explains the student's errors) |

## Files needed

Put the three folders side by side; the scripts read `data/`, use `tools/` and write `outputs/` next to them:

```
TRACE-SR/
├── data/        ProgrammableWeb/, APIBench_Q/java/, APIBench_Q/python/
├── generate/    these scripts
└── tools/       reading data, mapping model output to the API catalog, evaluation (see tools/README.md)
```

## Setup

```bash
pip install -r generate/requirements.txt
```

- Student scripts need one CUDA GPU (about 20 GB of memory at the default batch size of 32). The model is loaded from Hugging Face (`meta-llama/Llama-3.2-3B-Instruct`): accept the Llama 3.2 license on its model page, then run `huggingface-cli login`. If the model is already downloaded, add `--model /path/to/Llama-3.2-3B-Instruct` to every student command instead.
- GPT scripts need no GPU, only an OpenAI key: `export OPENAI_API_KEY=...`

## Run

Datasets: `programmableweb`, `apibench_java`, `apibench_python`. Run step 1 first; the error scripts read its output.

```bash
python generate/recommend.py          --dataset programmableweb   # step 1 -> outputs/programmableweb/predictions_train.jsonl
python generate/explain_gold.py       --dataset programmableweb   # C      -> outputs/programmableweb/gold_explanations_train.jsonl
python generate/explain_errors.py     --dataset programmableweb   # E      -> outputs/programmableweb/error_explanations_train.jsonl
python generate/gpt_explain_gold.py   --dataset programmableweb   # B      -> outputs/programmableweb/gpt-6-luna-none/gold_explanations_train.jsonl
python generate/gpt_explain_errors.py --dataset programmableweb   # D      -> outputs/programmableweb/gpt-6-luna-none/error_explanations_train.jsonl
```

| File | Content |
| --- | --- |
| `predictions_train.jsonl` | `id`, `gold`, raw `output`, parsed `predicted`: the names the model wrote, each with the `api_id` it refers to (null for names not in the dataset). The rules are in `tools/mapping.py` |
| `gold_explanations_train.jsonl` | `id`, `gold`, `explanation` |
| `error_explanations_train.jsonl` | `id`, `gold`, `wrong`, `missed`, `explanation` (only samples with at least one error; another label of the same API, such as `Sorted` for `sorted`, is not an error) |
| `*.log.jsonl` | run time, settings, and GPU and library versions (student) or the served model snapshot (GPT) |

Every record also has `prompt_tokens`, `completion_tokens` and `truncated` (the output hit `--max-new-tokens`); GPT records also have `reasoning_tokens`.

## GPT settings

Defaults: `gpt-6-luna`, reasoning effort `none`, and the same prompts and output limit as the student (768 tokens). GPT results go to a folder named `<model>-<effort>`, so different settings never mix. To try more reasoning, raise the output limit as well, since reasoning tokens count toward it:

```bash
python generate/gpt_explain_gold.py --dataset programmableweb --reasoning-effort medium --max-new-tokens 4096 --limit 100
```

`--workers` (default 8) sets the number of parallel requests; rate-limited requests are retried.

## Options

- `--split` (default `train`); `--limit N` runs only the first N samples, for a quick test.
- `--batch-size` (default 32): lower it if the GPU runs out of memory.
- `--max-new-tokens`: 128 for step 1, 768 for the explanations.
- Choose the GPU with `CUDA_VISIBLE_DEVICES`, e.g. `CUDA_VISIBLE_DEVICES=1 python generate/recommend.py --dataset programmableweb`.
- A stopped run can be restarted with the same command; finished samples are skipped, and the rows end up in sample order.
- After the mapping rules change, `python tools/remap.py --dataset ...` re-maps the saved step-1 outputs and drops the error explanations whose input changed; then rerun `explain_errors.py` and `gpt_explain_errors.py` to fill them in.

## Reproducibility

The student's prompt carries a fixed date ("Today Date: 26 Jul 2024"); Llama 3.2's chat template would otherwise insert the day of the run. The student decodes greedily, but bf16 arithmetic still differs slightly across GPUs, library versions and batch sizes, which can change an occasional word and everything after it. Use the same batch size and similar library versions to reproduce a run. GPT outputs are not fully deterministic; the served model snapshot is recorded in the log.
