"""LoRA fine-tuning of the student (Llama-3.2-3B-Instruct) for one condition of TRACE_SR.pdf on one dataset.

python train/sft.py --dataset programmableweb --condition D

The main task and, for B-E, the explanation task (train/examples.py) are mixed and shuffled; the loss counts only
the output tokens. After every epoch the LoRA weights are saved and the model is scored on dev, the way
train/predict.py scores it. Writes
  checkpoints/<dataset>/<run>/  config.json, train_log.jsonl, epoch1/ epoch2/ ..., best.json (best dev F1)
  results/<dataset>/<run>/      predictions_dev_epoch<N>.jsonl, metrics_dev_epoch<N>.json
Defaults: LoRA and optimizer from torchtune's Llama-3.2-3B LoRA recipe; data mixing, output-only loss and greedy
decoding from LEMA; three epochs with the best one chosen on dev, as in SelfDR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import examples  # noqa: E402  (puts generate/ and the project root on sys.path)
import predict  # noqa: E402
import common  # noqa: E402
from tools import data  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # every linear layer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=data.DATASETS)
    parser.add_argument("--condition", required=True, choices=examples.RUNS)
    parser.add_argument("--model", default=common.STUDENT, help="Hugging Face id or local folder of the base model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8, help="examples per forward pass")
    parser.add_argument("--grad-accum", type=int, default=4, help="passes per update: 8 x 4 = 32 examples per update")
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=float, default=0.05, help="fraction of the updates")
    parser.add_argument("--max-len", type=int, default=1024, help="prompt + output tokens; longer examples are dropped")
    parser.add_argument("--explanation-tokens", type=int, default=384, help="explanations are cut to this many tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, help="only the first N training and dev samples (for a quick test)")
    parser.add_argument("--no-eval", action="store_true", help="do not score dev after each epoch")
    return parser.parse_args()


def tokenize(tok, prompt, target):
    """Token ids of the chat (as at inference) and labels that are -100 everywhere except the output."""
    user = [{"role": "user", "content": prompt}]
    head = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True, date_string=common.DATE)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": target}], tokenize=False, date_string=common.DATE)
    assert full.startswith(head), "the chat template does not put the answer after the prompt"
    head_ids = tok(head, add_special_tokens=False).input_ids
    tail_ids = tok(full[len(head):], add_special_tokens=False).input_ids  # the output and its end-of-turn token
    return head_ids + tail_ids, [-100] * len(head_ids) + tail_ids


def batches(items, batch_size, pad_id, generator):
    """Shuffled, right-padded batches of (input ids, labels, attention mask)."""
    order = torch.randperm(len(items), generator=generator).tolist()
    for i in range(0, len(order), batch_size):
        chunk = [items[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        ids = torch.full((len(chunk), width), pad_id)
        labels = torch.full((len(chunk), width), -100)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, (x, y) in enumerate(chunk):
            ids[k, :len(x)], labels[k, :len(y)], mask[k, :len(x)] = torch.tensor(x), torch.tensor(y), 1
        yield ids, labels, mask


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run = examples.RUNS[args.condition]
    checkpoint, results = predict.CHECKPOINTS / args.dataset / run, predict.RESULTS / args.dataset / run
    checkpoint.mkdir(parents=True, exist_ok=True)

    tok, net = predict.load_model(args.model)
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    from peft import LoraConfig, get_peft_model
    net = get_peft_model(net, LoraConfig(task_type="CAUSAL_LM", r=args.lora_r, lora_alpha=args.lora_alpha,
                                         lora_dropout=args.lora_dropout, target_modules=LORA_TARGETS))
    trainable = [p for p in net.parameters() if p.requires_grad]
    for p in trainable:  # LoRA weights train in fp32; the frozen model stays in bf16
        p.data = p.data.float()

    rows, counts = examples.build(args.dataset, args.condition, tok, args.explanation_tokens, args.limit)
    items = [tokenize(tok, r["prompt"], r["target"]) for r in rows]
    counts["dropped_too_long"] = sum(len(x) > args.max_len for x, _ in items)
    items = [(x, y) for x, y in items if len(x) <= args.max_len]
    updates_per_epoch = math.ceil(math.ceil(len(items) / args.batch_size) / args.grad_accum)
    total = updates_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(args.warmup * total), total)

    split_folder = data.ROOT / "data" / data.DATASETS[args.dataset]
    inputs = [split_folder / "train.jsonl", split_folder / "dev.jsonl", split_folder / "apis.jsonl"]
    if args.condition != "A":
        inputs.append(data.output(args.dataset, examples.EXPLANATIONS[args.condition]))
    if args.condition in "DE":
        inputs.append(data.output(args.dataset, "predictions_train.jsonl"))
    code = [Path(__file__).parent / f for f in ("examples.py", "sft.py", "predict.py")] + [data.ROOT / "generate" / f for f in ("common.py", "recommend.py")]
    import peft
    import transformers
    config = {"dataset": args.dataset, "condition": args.condition, "run": run, "args": vars(args), "lora_targets": LORA_TARGETS,
              "examples": counts, "updates_per_epoch": updates_per_epoch, "updates": total,
              "inputs": {str(p.relative_to(data.ROOT)): sha256(p) for p in inputs},
              "code": {str(p.relative_to(data.ROOT)): sha256(p) for p in code},
              "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"{run}: {counts}; {total} updates")

    log = open(checkpoint / "train_log.jsonl", "w", encoding="utf-8", newline="\n")
    generator, start, step, tokens, losses, dev = torch.Generator().manual_seed(args.seed), time.time(), 0, 0, [], {}
    for epoch in range(1, args.epochs + 1):
        net.train()
        passes = list(batches(items, args.batch_size, tok.pad_token_id, generator))
        for i, (ids, labels, mask) in enumerate(passes, 1):
            with torch.autocast(predict.DEVICE, dtype=torch.bfloat16):
                loss = net(input_ids=ids.to(predict.DEVICE), attention_mask=mask.to(predict.DEVICE), labels=labels.to(predict.DEVICE)).loss
            (loss / args.grad_accum).backward()
            losses.append(loss.item())
            tokens += int((labels != -100).sum())
            if i % args.grad_accum and i < len(passes):  # keep accumulating; the last passes of an epoch always update
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == total:
                log.write(json.dumps({"step": step, "epoch": epoch, "loss": round(sum(losses) / len(losses), 4),
                                      "lr": scheduler.get_last_lr()[0], "output_tokens": tokens,
                                      "seconds": round(time.time() - start)}) + "\n")
                log.flush()
                losses = []
        net.save_pretrained(checkpoint / f"epoch{epoch}")
        if not args.no_eval:
            net.eval()
            dev[epoch] = predict.evaluate(net, tok, args.dataset, "dev", results, f"dev_epoch{epoch}", epoch, args.limit)
            best = max(dev, key=lambda e: dev[e]["f1"])
            (checkpoint / "best.json").write_text(json.dumps({"best_epoch": best, "metric": "dev f1", "dev": dev}, indent=2) + "\n", encoding="utf-8")
            print(f"epoch {epoch}: dev f1 {dev[epoch]['f1']:.4f} (best: epoch {best})")
    log.close()
    config.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=round(time.time() - start), output_tokens=tokens)
    (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
