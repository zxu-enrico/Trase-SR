"""Regression tests for mapping written names to the catalog and for comparing with the gold set.

python -m unittest discover -s tools/tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.evaluate import compare, scores  # noqa: E402
from tools.mapping import Catalog  # noqa: E402


def catalog(*entries):
    """entries: name or (name, documented_as); api_id = name."""
    apis = {}
    for entry in entries:
        name, documented = entry if isinstance(entry, tuple) else (entry, None)
        apis[name] = {"api_id": name, "name": name, **({"documented_as": documented} if documented else {})}
    return Catalog(apis)


def resolved(cat, name):
    api = cat.resolve(name)
    return api["api_id"] if api else None


class JavaCodeForms(unittest.TestCase):
    cat = catalog("java.lang.String.substring()", "java.util.Collections.swap()", "java.sql.Connection.setAutoCommit()",
                  "java.io.File.File()", "java.util.Calendar.add()", "java.lang.String.split()",
                  ("java.lang.String.solit()", "java.lang.String.split"),
                  ("File(path).getParentFile().getName()", "java.io.File"), ("Scanner()", "java.util.Scanner"))

    def test_arguments_and_separators(self):
        for name in ["String.substring(int beginIndex)", "String.substring(int beginIndex, int endIndex)",
                     "java.lang.String.substring(int, int);"]:
            self.assertEqual(resolved(self.cat, name), "java.lang.String.substring()")
        self.assertEqual(resolved(self.cat, "java.util.Collections#swap(List, int, int)"), "java.util.Collections.swap()")
        self.assertEqual(resolved(self.cat, "java.sql.Connection.setAutoCommit(boolean)"), "java.sql.Connection.setAutoCommit()")
        self.assertEqual(resolved(self.cat, "java.util.Calendar.add(Calendar.DAY_OF_MONTH, -1)"), "java.util.Calendar.add()")

    def test_overloads_are_one_prediction(self):
        parsed = self.cat.parse("String.substring(int beginIndex)\nString.substring(int beginIndex, int endIndex)")
        self.assertEqual([p["api_id"] for p in parsed], ["java.lang.String.substring()"])

    def test_constructor(self):
        self.assertEqual(resolved(self.cat, "new File(path)"), "java.io.File.File()")
        self.assertEqual(resolved(self.cat, "java.util.Scanner"), "Scanner()")  # the label documents the class itself

    def test_conservative(self):
        self.assertIsNone(resolved(self.cat, "java.io.File"))  # a class is not the call chain its class documents
        self.assertIsNone(resolved(self.cat, "substring"))  # a bare method name is not guessed
        self.assertIsNone(resolved(self.cat, "org.other.String.substring"))  # another package
        ambiguous = catalog("java.util.List.add()", "java.awt.List.add()")
        self.assertIsNone(resolved(ambiguous, "List.add"))

    def test_typo_label_is_the_same_api(self):
        self.assertEqual(self.cat.identity(self.cat.apis["java.lang.String.solit()"]),
                         self.cat.identity(self.cat.apis["java.lang.String.split()"]))


class PythonForms(unittest.TestCase):
    cat = catalog(("reverse", "list.reverse"), ("prod", "math.prod"), "logging.basicConfig", "os.system", "pandas.concat",
                  ("Sorted", "sorted"), "sorted", "os.path.join", "numpy.random.random")

    def test_documented_api_is_a_spelling(self):
        self.assertEqual(resolved(self.cat, "list.reverse()"), "reverse")
        self.assertEqual(resolved(self.cat, "math.prod"), "prod")

    def test_calls(self):
        self.assertEqual(resolved(self.cat, "logging.basicConfig(level=logging.INFO)"), "logging.basicConfig")
        parsed = self.cat.parse("os.system('cls')\nos.system('clear')\npandas.concat\npandas.concat([s for s in series], axis=0)")
        self.assertEqual([p["api_id"] for p in parsed], ["os.system", "pandas.concat"])

    def test_exact_case_wins(self):
        self.assertEqual(resolved(self.cat, "sorted"), "sorted")
        self.assertEqual(resolved(self.cat, "Sorted"), "Sorted")

    def test_imports(self):
        cat = catalog(("Counter", "collections.Counter"), "os.system")
        self.assertEqual(resolved(cat, "from collections import Counter"), "Counter")
        self.assertIsNone(resolved(cat, "import os"))  # a module is not one of its functions

    def test_module_paths_are_not_shortened(self):
        self.assertIsNone(resolved(self.cat, "path.join"))
        self.assertIsNone(resolved(self.cat, "random.random"))


class WebAPIs(unittest.TestCase):
    cat = catalog("Twitter", "Google Maps", "Zappos", "The Movie Database (TMDb)", "Pixabay - Free Images")

    def test_suffix_and_notes(self):
        self.assertEqual(resolved(self.cat, "Zappos API"), "Zappos")
        self.assertEqual(resolved(self.cat, "Twitter REST API"), "Twitter")
        self.assertEqual(resolved(self.cat, "Google Maps (for venue location)"), "Google Maps")
        self.assertEqual(resolved(self.cat, "Twitter API: posts tweets"), "Twitter")

    def test_names_with_punctuation_and_negated_notes(self):
        self.assertEqual(resolved(self.cat, "The Movie Database (TMDb)"), "The Movie Database (TMDb)")
        self.assertEqual(resolved(self.cat, "Pixabay - Free Images"), "Pixabay - Free Images")
        self.assertIsNone(resolved(self.cat, "Google Maps (not applicable)"))

    def test_parse_skips_chatter_and_fences(self):
        text = "```\nHere are the APIs:\n- Twitter API\n- Google Maps, and Zappos\nI hope this helps!\n```"
        self.assertEqual([p["api_id"] for p in self.cat.parse(text)], ["Twitter", "Google Maps", "Zappos"])


class Comparison(unittest.TestCase):
    cat = catalog(("Sorted", "sorted"), "sorted", "split", "sort")

    def test_another_label_of_a_gold_api_is_correct(self):
        predicted = [{"name": "Sorted", "api_id": "Sorted"}, {"name": "os.sort", "api_id": None}]
        self.assertEqual(compare(["sorted", "split"], predicted, self.cat), (["os.sort"], ["split"]))

    def test_scores(self):
        rows = [{"gold": ["sorted"], "predicted": [{"name": "Sorted", "api_id": "Sorted"}]},
                {"gold": ["split", "sort"], "predicted": [{"name": "split", "api_id": "split"}, {"name": "x", "api_id": None}]}]
        result = scores(rows, self.cat)
        self.assertEqual((result["precision"], result["recall"], result["exact"]), (2 / 3, 2 / 3, 0.5))


if __name__ == "__main__":
    unittest.main()
