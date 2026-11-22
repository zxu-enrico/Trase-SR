# tools

Code shared by `generate/` and the training code: reading the datasets, turning a model's answer into catalog APIs, and scoring it. Nothing here calls a model.

| File | What it does |
| --- | --- |
| `data.py` | where the datasets and outputs are; reading and writing JSONL |
| `mapping.py` | turns the API names a model writes into catalog APIs, and decides when two catalog labels are the same API |
| `evaluate.py` | wrong and missed APIs of a prediction; precision, recall, F1 and exact-set accuracy |
| `remap.py` | applies changed mapping rules to the saved step-1 outputs and drops the error explanations whose input changed |
| `tests/` | regression tests for the mapping |

```bash
python tools/evaluate.py --dataset programmableweb      # scores outputs/programmableweb/predictions_train.jsonl
python tools/remap.py --dataset programmableweb --dry-run   # what a mapping change would alter
python -m unittest discover -s tools/tests
```

## Mapping rules

The gold set is never used to read a name. A name is tried as written, then as code, then without "API", then without a trailing description; the first form found in the catalog wins.

| A model writes | It counts as | Rule |
| --- | --- | --- |
| `sorted` | `sorted`, not `Sorted` | exact spelling before ignoring case |
| `math.prod`, `list.reverse()` | `prod`, `reverse` | a label's documented API (`documented_as`) is another spelling of it |
| `String.substring(int beginIndex)`, `java.util.Collections#swap(List, int, int)`, `os.system('cls')` | `java.lang.String.substring()`, `java.util.Collections.swap()`, `os.system` | arguments, generics, `#`, `::`, `new` and `;` are dropped; `Class.method` matches when one API ends that way |
| `new File(path)` | `java.io.File.File()` | a constructor call is the constructor label |
| `from collections import Counter` | `Counter` (documented as `collections.Counter`) | an import names what it imports |
| `Twitter API`, `Google Maps (for venue location)` | `Twitter`, `Google Maps` | a trailing "API" and a trailing description are dropped |
| `java.util.List`, `substring`, `path.join`, `Google Maps (not applicable)` | not in the catalog | a class or module alone is not one of its methods; bare method names and shortened module paths are not guessed; a negated note is no recommendation |

Labels of the same API (the same documented API, or the same name up to case and spacing) are one API: a prediction lists it once, and it matches any of its labels in the gold set. A name that is not in the catalog counts as a wrong recommendation.
