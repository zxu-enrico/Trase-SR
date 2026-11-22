"""Shared pieces of the generation steps: task wording, the student and GPT models, and the generation loop.
Dataset files, mapping model output to the API catalog and evaluation are in tools/ (next to generate/)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the project root, for tools/
from tools.data import DATASETS, ROOT, load, output, read_jsonl, write_jsonl  # noqa: E402,F401  (used by the scripts)

STUDENT = "meta-llama/Llama-3.2-3B-Instruct"  # Hugging Face id; pass --model <folder> to use a local copy
GPT_MODEL = "gpt-6-luna"
# Llama 3.2's chat template otherwise writes today's date into the prompt ("Today Date: ..."), so a run on another day
# would see a different prompt. "26 Jul 2024" is the template's own default.
DATE = "26 Jul 2024"
# What the request is, what is recommended, and how an API name is written in that dataset.
TASKS = {
    "programmableweb": ("mashup description", "Web APIs", "Google Maps"),
    "apibench_java": ("Java programming question", "Java API methods", "java.io.File.mkdirs()"),
    "apibench_python": ("Python programming question", "Python APIs", "os.path.join"),
}


def parse_args(doc, max_new_tokens, gpt=False):
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    if gpt:
        parser.add_argument("--model", default=GPT_MODEL, help="OpenAI model name")
        parser.add_argument("--reasoning-effort", default="none", choices=["none", "low", "medium", "high", "xhigh", "max"])
        parser.add_argument("--workers", type=int, default=8, help="parallel API requests")
    else:
        parser.add_argument("--model", default=STUDENT, help="Hugging Face id or local folder of the student model")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, help="only the first N samples (for a quick test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    return parser.parse_args()


def doc(api, limit=600):
    text = " ".join(api["description"].split())
    return f"- {api['name']}: {text[:limit]}{'...' if len(text) > limit else ''}"


def greedy(net, tok, prompts, max_new_tokens):
    """[(text, usage)]: greedy completions of a batch of user prompts with the chat template and the fixed date.
    `tok` must pad on the left. Also used by train/ to evaluate fine-tuned models in the same way."""
    import torch
    chats = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                     date_string=DATE) for p in prompts]
    batch = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(net.device)
    with torch.no_grad():
        # Greedy decoding; clear the sampling defaults (temperature 0.6, top_p 0.9) from the model's generation_config.
        out = net.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                           pad_token_id=tok.pad_token_id)
    results = []
    for row, mask in zip(out[:, batch["input_ids"].shape[1]:], batch["attention_mask"]):
        completion = int((row != tok.pad_token_id).sum())
        results.append((tok.decode(row, skip_special_tokens=True).strip(),
                        {"prompt_tokens": int(mask.sum()), "completion_tokens": completion, "truncated": completion >= max_new_tokens}))
    return results


class Student:
    """The local student model (Hugging Face), greedy decoding on one GPU."""

    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.net = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
        self.max_new_tokens = args.max_new_tokens

    def __call__(self, prompts):
        """[(text, usage)] for a batch of prompts."""
        return greedy(self.net, self.tok, prompts, self.max_new_tokens)

    def describe(self):
        import torch
        import transformers
        return {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
                "torch": torch.__version__, "transformers": transformers.__version__}


class GPT:
    """An OpenAI model through the Responses API. Reads OPENAI_API_KEY (and OPENAI_BASE_URL for a compatible endpoint)."""

    def __init__(self, args):
        import openai
        self.openai, self.client = openai, openai.OpenAI()
        self.model, self.effort, self.max_tokens, self.workers = args.model, args.reasoning_effort, args.max_new_tokens, args.workers
        self.served = None
        if self.effort != "none" and self.max_tokens < 2048:
            print(f"note: reasoning tokens count toward --max-new-tokens ({self.max_tokens}); raise it for effort '{self.effort}'")

    def __call__(self, prompts):
        """[(text, usage)] for a batch of prompts, sent in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.ask, prompts))

    def ask(self, prompt):
        retry = (self.openai.RateLimitError, self.openai.APIConnectionError, self.openai.APITimeoutError, self.openai.InternalServerError)
        for attempt in range(8):
            try:
                r = self.client.responses.create(model=self.model, input=prompt, reasoning={"effort": self.effort},
                                                 max_output_tokens=self.max_tokens)
                break
            except retry:
                time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"requests to {self.model} kept failing")
        self.served = r.model
        details = getattr(r.usage, "output_tokens_details", None)
        return r.output_text.strip(), {"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens,
                                       "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
                                       "truncated": r.status == "incomplete"}

    def describe(self):
        return {"reasoning_effort": self.effort, "served_model": self.served, "openai": self.openai.__version__}


def run(args, samples, path, prompt, record, backend=None):
    """Generate for the samples not yet in `path` (runs can be resumed) and append the results. backend: Student or GPT."""
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [s for s in samples[:args.limit] if s["id"] not in done]
    if not todo:
        return print(f"nothing to do: {path}")
    model = (backend or Student)(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(path, "a", encoding="utf-8", newline="\n") as f:  # LF on every OS, so files compare across machines
        for i in range(0, len(todo), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            for sample, (text, usage) in zip(chunk, model([prompt(s) for s in chunk])):
                f.write(json.dumps({**record(sample, text), **usage}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{min(i + args.batch_size, len(todo))}/{len(todo)}", flush=True)
    with open(path.with_suffix(".log.jsonl"), "a", encoding="utf-8", newline="\n") as f:  # time and settings, for cost and reproducibility
        f.write(json.dumps({"model": args.model, "split": args.split, "samples": len(todo), "seconds": round(time.time() - start, 1),
                            "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size, **model.describe(),
                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    if done:  # a resumed run appended its rows at the end: put them back in sample order
        position = {s["id"]: i for i, s in enumerate(samples)}
        write_jsonl(path, sorted(read_jsonl(path), key=lambda r: position.get(r["id"], len(position))))
    print(f"wrote {path}")
