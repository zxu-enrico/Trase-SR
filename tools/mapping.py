"""Map the API names a model writes to the dataset's API catalog, and tell when two catalog labels are the same API.

The gold set is never consulted. A written name is tried in these forms, and the first one found in the catalog wins:

1. As written: exact spelling first, then ignoring case if that does not mix up two APIs ("sorted" is `sorted`, not
   `Sorted`). A label's documented API (`documented_as`) is also a spelling of it ("math.prod" finds the label `prod`),
   unless a class documents a method or call chain.
2. As code: arguments, generics, "new" and a trailing ";" removed, "#" and "::" read as ".", an import read as the
   name it imports ("String.substring(int beginIndex)", "java.util.Collections#swap(List, int, int)",
   "os.system('cls')", "from collections import Counter").
   A constructor call ("new File(path)") is also tried as the dataset's constructor label `File.File`.
3. Without a trailing "API", "Web API", "REST API" or "Web Service" ("Twitter API" is `Twitter`).
4. Without a trailing description: "name: what it does", "name - what it does", "name (what it does)". A negated
   note is not a recommendation: "Google Maps (not applicable)" stays unmatched.

In every form, "Class.method" also matches a fully qualified label when exactly one API ends that way
("String.substring" is `java.lang.String.substring`). A class or module alone never stands for one of its methods, a
bare method name only matches a label written that way, and a name with another package does not match.

Labels of the same API (the same documented API, or the same name up to case and spacing) are one API: a prediction
lists it once, and it matches any of its labels in the gold set.
"""

from __future__ import annotations

import re
from collections import defaultdict

IDENTIFIER = r"[A-Za-z_$][\w$]*"
DOTTED = re.compile(rf"{IDENTIFIER}(?:\.{IDENTIFIER})*")
CLASS_METHOD = re.compile(rf"[A-Z][\w$]*\.{IDENTIFIER}")
API_SUFFIX = re.compile(r"(?i)\s+(?:(?:(?:web|rest|restful)\s+)?apis?|(?:web|rest|restful)\s+services?)$")
# "name: what it does", "name - what it does", "name (what it does)"; a negated note ("not applicable") is no recommendation
DESCRIPTION = [re.compile(r"^(.+?):\s+(\S.*)$"), re.compile(r"^(.+?)\s+[-–—]\s+(\S.*)$"), re.compile(r"^(.*?\S)\s*\(([^()]*)\)$")]
NEGATION = re.compile(r"(?i)\s*(?:not|no|none|n/?a|never|cannot|can't|isn't|doesn't)\b")
FENCE = re.compile(r"\s*(?:`{3,}|~{3,})[A-Za-z0-9_+.-]*\s*")
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
# A line where the model talks instead of naming APIs: refusals, notes, follow-up questions.
CHATTER = re.compile(r"^(?:i\b|i['’]m\b|sorry\b|unfortunately\b|please\b|if you\b|is there\b|there (?:is|are)\b|as an ai\b|"
                     r"note\b|here (?:is|are)\b|these\b|this\b|it(?:['’]s| is)\b|but\b|without\b|based on\b|however\b|"
                     r"to (?:implement|achieve|do|build|create|use)\b|you (?:can|could|may|might|would|will|need)\b|"
                     r"alternatively\b|additionally\b|also\b|for example\b|in addition\b|the following\b)", re.I)


def key(name):
    """Spelling key: ignores case, spaces, backticks and parentheses."""
    return _spelling(name).casefold()


def _spelling(name):
    return re.sub(r"[\s`()]+", "", name)


def _clean(text):
    return text.strip().strip("`'\"*").strip().rstrip(".,;").strip()


def _member(spelling):
    parts = [p for p in re.split(r"\.|#|::", _spelling(spelling)) if p]
    return parts[-1] if parts else ""


def _documented(api):
    """The label's documented API, unless that is a class documenting a method or call chain: the chain
    `File(path).getParentFile().getName()` is documented by the class java.io.File, but it is not the class."""
    documented = api.get("documented_as")
    if documented and _member(documented)[:1].isupper() and not _member(api["name"])[:1].isupper():
        return None
    return documented


def _code_forms(name):
    """The name as a code reference: "java.util.Collections#swap(List, int, int)" -> java.util.Collections.swap."""
    text = name.strip().rstrip(";").strip()
    imported = re.fullmatch(r"(?:from\s+([\w.]+)\s+)?import\s+([\w.]+)(?:\s+as\s+\w+)?", text)
    if imported:  # "from collections import Counter" names collections.Counter; "import os" names only a module
        text = ".".join(filter(None, imported.groups()))
    new = re.match(r"new\s+", text) is not None
    text = re.sub(r"^new\s+", "", text).replace("#", ".").replace("::", ".")
    while re.search(r"<[^<>]*>", text):
        text = re.sub(r"<[^<>]*>", "", text)
    call = False
    if text.endswith(")"):  # drop the argument list that closes the name; "MySpace (not available)" is a note, not one
        depth = 0
        for i in range(len(text) - 1, -1, -1):
            depth += (text[i] == ")") - (text[i] == "(")
            if depth == 0:
                if i > 0 and not text[i - 1].isspace():
                    text, call = text[:i], True
                break
    if not DOTTED.fullmatch(text):
        return []
    last = text.rsplit(".", 1)[-1]
    constructor = [f"{text}.{last}"] if (new or call) and last[:1].isupper() else []
    return constructor + [text]


def _split_names(line):
    """Comma-separated names on one line ("a, b, and c"); commas inside (), [], <> or {} do not split."""
    parts, depth, current = [], 0, ""
    for ch in line:
        depth = depth + (ch in "([<{") - (ch in ")]>}" and depth > 0)
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    return [re.sub(r"^(?:and|or)\s+", "", p.strip()) for p in parts + [current] if p.strip()]


class Catalog:
    """A dataset's APIs (apis.jsonl, keyed by api_id), indexed to resolve written names."""

    def __init__(self, apis):
        self.apis = apis
        self._exact, folded, tails = {}, defaultdict(list), defaultdict(list)
        spellings = [(a["name"], a) for a in apis.values()]
        spellings += [(_documented(a), a) for a in apis.values() if _documented(a)]
        for spelling, api in spellings:  # labels first, so a label wins over another label's documented API
            self._exact.setdefault(_spelling(spelling), api)
            folded[key(spelling)].append(api)
            parts = _spelling(spelling).replace("#", ".").replace("::", ".").split(".")
            if len(parts) >= 3 and parts[-2][:1].isupper():
                tails[key(".".join(parts[-2:]))].append(api)
        self._folded = {k: v[0] for k, v in folded.items() if len({self.identity(a) for a in v}) == 1}
        self._tails = {k: v[0] for k, v in tails.items() if len({self.identity(a) for a in v}) == 1}

    @staticmethod
    def identity(api):
        """The API a label denotes, as a key; labels with the same identity are the same API."""
        return key(_documented(api) or api["name"]).lstrip(".")

    def _lookup(self, spelling):
        api = self._exact.get(_spelling(spelling)) or self._folded.get(key(spelling))
        if api is None and CLASS_METHOD.fullmatch(_spelling(spelling)):
            api = self._tails.get(key(spelling))
        return api

    def resolve(self, name):
        """The catalog entry (a row of apis.jsonl) that a written name refers to, or None."""
        heads = [m.group(1) for pattern in DESCRIPTION if (m := pattern.match(name)) and not NEGATION.match(m.group(2))]
        forms = [name, *_code_forms(name), API_SUFFIX.sub("", name)]
        for head in heads:
            forms += [head, *_code_forms(head), API_SUFFIX.sub("", head)]
        for form in dict.fromkeys(f.strip() for f in forms if f.strip()):
            api = self._lookup(form)
            if api is not None:
                return api
        return None

    def parse(self, text):
        """API names written one per line or comma-separated -> [{"name", "api_id"}]; api_id is None for a name that is
        not in the catalog. Lines where the model talks instead of naming APIs (refusals, notes, questions) and code
        fences are skipped, and an API that is already listed is not listed again."""
        found, seen = [], set()
        for line in text.splitlines():
            if FENCE.fullmatch(line):
                continue
            line = _clean(BULLET.sub("", line))
            if not line or line.endswith(":"):
                continue
            if self.resolve(line):  # an API, even if its name has a comma
                names = [line]
            elif CHATTER.match(line) or line.endswith("?"):
                continue
            else:
                names = [_clean(n) for n in _split_names(line)]
            for name in names:
                if not name or name.endswith(":"):
                    continue
                api = self.resolve(name)
                identity = self.identity(api) if api else key(name)
                if identity not in seen:
                    seen.add(identity)
                    found.append({"name": name, "api_id": api["api_id"] if api else None})
        return found
