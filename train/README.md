# train

LoRA fine-tuning of the student (Llama-3.2-3B-Instruct) under the five conditions of `TRACE_SR.pdf`, one dataset at a time, and scoring on dev/test. At inference every condition gets only the request (the step-1 prompt of `generate/recommend.py`) and answers with API names.

| Condition | Run name | Main task (every training sample) | Explanation task (samples whose draft has an error) |
| --- | --- | --- | --- |
| A | `A-labels` | request → correct APIs | none |
| B | `B-gpt-gold` | request → correct APIs | request → GPT correct-answer explanation + correct APIs |
| C | `C-student-gold` | request → correct APIs | request → student correct-answer explanation + correct APIs |
| D | `D-gpt-error` | request → correct APIs | request + student draft → GPT error explanation + correct APIs |
| E | `E-student-error` | request → correct APIs | request + student draft → student error explanation + correct APIs |

B–E use the same samples for the explanation task, and every explanation is cut at its last whole line within 384 tokens (`--explanation-tokens`). At 384, at most 7% of the GPT error explanations are cut; for the student's longer explanations the cut mostly removes rambling or repeated tails. The drafts and explanations are read from `outputs/<dataset>/` (see `generate/`).

```bash
pip install -r train/requirements.txt
python train/sft.py     --dataset programmableweb --condition D              # train; scores dev after every epoch
python train/predict.py --dataset programmableweb --condition D --split test # test with the best dev epoch
python train/predict.py --dataset programmableweb --split test               # base model, zero-shot
```

Datasets: `programmableweb`, `apibench_java`, `apibench_python`. `--model /path/to/Llama-3.2-3B-Instruct` uses a local copy; `--limit N` uses only the first N samples, for a quick test.

## Settings

| Setting | Value | Source |
| --- | --- | --- |
| LoRA | r 64, alpha 128, on every linear layer | torchtune Llama-3.2-3B LoRA recipe; all linear layers as in LEMA and WAR-R1 |
| LoRA dropout | 0.05 | LEMA |
| Optimizer | AdamW, lr 3e-4, weight decay 0.01, bf16 model with fp32 LoRA weights, gradient checkpointing | torchtune |
| Schedule | cosine, warmup 5% of the updates, gradient clipping 1.0 | torchtune (100 warmup steps ≈ 6% of its run) |
| Batch | 32 examples per update (8 × 4 accumulation) | torchtune |
| Epochs | 3; the epoch with the best dev F1 is kept for test | SelfDR (3 epochs, early stopping) |
| Data | main and explanation examples mixed and shuffled; loss on output tokens only | LEMA |
| Length | at most 1,024 tokens per example | LEMA (512 input + 512 output) |
| Decoding | greedy, 128 new tokens | LEMA (greedy); step 1 (128) |
| Seed | 42 | |

## Files

```
checkpoints/<dataset>/<run>/
├── config.json        settings, example counts, explanations cut, SHA-256 of the inputs and the code
├── train_log.jsonl    loss, learning rate, output tokens and time every 10 updates
├── epoch1/ epoch2/ epoch3/    LoRA weights (adapter_model.safetensors, adapter_config.json), about 0.4 GB each
└── best.json          the epoch with the best dev F1, and the dev metrics of every epoch
results/<dataset>/<run>/       predictions_dev_epoch<N>.jsonl, metrics_dev_epoch<N>.json, predictions_test.jsonl, metrics_test.json
results/<dataset>/zeroshot/    the base model
```

Metrics (`tools/evaluate.py`): precision, recall and F1 (primary), macro F1, exact-set accuracy, and the wrong (in or outside the catalog) and missed APIs.
