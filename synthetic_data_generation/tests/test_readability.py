# This file checks that every Python file in the synthetic data package is easy
# to read for someone who has never used Python. It reads each file as text and
# as a syntax tree (the shape Python itself builds from the code) and reports
# every place where a readability rule is broken. The rules are: a plain
# comment on the first line; a docstring (a short description in quotes) on every
# function and class; and none of the harder-to-read features: lambda, nested
# comprehensions, decorators other than a plain dataclass, *args, **kwargs and
# star-unpacking in a call. The file also proves each rule on small throwaway
# sample files, so removing any one check makes a test fail.

import ast
from pathlib import Path

# The package folder is found from this file's own location, so files added by
# later work are checked without editing this test.
PACKAGE_FOLDER = Path(__file__).resolve().parent.parent

COMPREHENSION_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
DEFINITION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def imports_dataclass_from_dataclasses(tree):
    """Say whether the file runs "from dataclasses import dataclass" without renaming it."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "dataclasses" or node.level != 0:
            continue
        for alias in node.names:
            if alias.name == "dataclass" and alias.asname is None:
                return True
    return False


def decorator_is_allowed(decorator, dataclass_imported):
    """Say whether a decorator is a plain dataclass, bare or with keyword arguments only."""
    if not dataclass_imported:
        return False
    if isinstance(decorator, ast.Name):
        return decorator.id == "dataclass"
    if isinstance(decorator, ast.Call):
        if not isinstance(decorator.func, ast.Name) or decorator.func.id != "dataclass":
            return False
        if len(decorator.args) > 0:
            return False
        for keyword in decorator.keywords:
            if keyword.arg is None:
                return False
        return True
    return False


def check_opening_comment(text):
    """Return rule failures when the first line is not a plain "#" comment."""
    first_line = text.split("\n", 1)[0]
    if first_line.startswith("#") and not first_line.startswith("#!"):
        return []
    return [("opening-comment", 1)]


def check_definition(node, dataclass_imported):
    """Return rule failures for one function or class: its docstring and its decorators."""
    failures = []
    docstring = ast.get_docstring(node)
    if docstring is None or docstring.strip() == "":
        failures.append(("docstring", node.lineno))
    for decorator in node.decorator_list:
        if not decorator_is_allowed(decorator, dataclass_imported):
            failures.append(("decorator", decorator.lineno))
    return failures


def check_comprehension(node):
    """Return rule failures for one comprehension: nesting and more than one "for"."""
    failures = []
    if len(node.generators) > 1:
        failures.append(("multiple-for-clauses", node.lineno))
    for inner in ast.walk(node):
        if inner is not node and isinstance(inner, COMPREHENSION_TYPES):
            failures.append(("nested-comprehension", node.lineno))
            break
    return failures


def check_parameters(node):
    """Return rule failures for one parameter list: *args and **kwargs (a bare * is fine)."""
    failures = []
    if node.vararg is not None:
        failures.append(("star-args", node.vararg.lineno))
    if node.kwarg is not None:
        failures.append(("star-kwargs", node.kwarg.lineno))
    return failures


def check_call(node):
    """Return rule failures for one call: star-unpacking of arguments."""
    failures = []
    for argument in node.args:
        if isinstance(argument, ast.Starred):
            failures.append(("call-unpacking", argument.lineno))
    for keyword in node.keywords:
        if keyword.arg is None:
            failures.append(("call-unpacking", keyword.value.lineno))
    return failures


def find_violations(path):
    """Return a list of (rule name, line number) for every rule the file at path breaks."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return [("does-not-parse", 1)]
    failures = check_opening_comment(text)
    dataclass_imported = imports_dataclass_from_dataclasses(tree)
    for node in ast.walk(tree):
        if isinstance(node, DEFINITION_TYPES):
            failures.extend(check_definition(node, dataclass_imported))
        if isinstance(node, ast.Lambda):
            failures.append(("lambda", node.lineno))
        if isinstance(node, COMPREHENSION_TYPES):
            failures.extend(check_comprehension(node))
        if isinstance(node, ast.arguments):
            failures.extend(check_parameters(node))
        if isinstance(node, ast.Call):
            failures.extend(check_call(node))
    return failures


def rule_names(failures):
    """Return the set of distinct rule names in a list of (rule name, line number)."""
    names = set()
    for rule, line in failures:
        names.add(rule)
    return names


OPENING = "# A sample file used to test the readability rules.\n"

# Each row is: label, the one rule the bad text breaks, the bad text, and a good
# text that passes. The texts are plain strings written to a temporary folder.
SAMPLES = [
    (
        "code-first",
        "opening-comment",
        'value = 1\n',
        OPENING + 'value = 1\n',
    ),
    (
        "module-docstring-instead-of-comment",
        "opening-comment",
        '"""A module docstring where a comment belongs."""\nvalue = 1\n',
        OPENING + 'value = 1\n',
    ),
    (
        "blank-first-line",
        "opening-comment",
        '\n' + OPENING + 'value = 1\n',
        OPENING + 'value = 1\n',
    ),
    (
        "shebang-first-line",
        "opening-comment",
        '#!/usr/bin/env python3\nvalue = 1\n',
        OPENING + 'value = 1\n',
    ),
    (
        "comment-only-init-file",
        "opening-comment",
        '',
        OPENING,
    ),
    (
        "function-without-docstring",
        "docstring",
        OPENING + 'def add(left, right):\n    return left + right\n',
        OPENING + 'def add(left, right):\n    "Add two numbers."\n    return left + right\n',
    ),
    (
        "function-with-empty-docstring",
        "docstring",
        OPENING + 'def add(left, right):\n    ""\n    return left + right\n',
        OPENING + 'def add(left, right):\n    "Add two numbers."\n    return left + right\n',
    ),
    (
        "async-function-without-docstring",
        "docstring",
        OPENING + 'async def fetch():\n    return 1\n',
        OPENING + 'async def fetch():\n    "Return one."\n    return 1\n',
    ),
    (
        "nested-function-without-docstring",
        "docstring",
        OPENING + 'def outer():\n    "Run."\n    def inner():\n        return 1\n    return inner\n',
        OPENING + 'def outer():\n    "Run."\n    def inner():\n        "Return one."\n        return 1\n    return inner\n',
    ),
    (
        "method-without-docstring",
        "docstring",
        OPENING + 'class Counter:\n    "Counts things."\n    def bump(self):\n        return 1\n',
        OPENING + 'class Counter:\n    "Counts things."\n    def bump(self):\n        "Return one."\n        return 1\n',
    ),
    (
        "class-without-docstring",
        "docstring",
        OPENING + 'class Counter:\n    value = 1\n',
        OPENING + 'class Counter:\n    "Counts things."\n    value = 1\n',
    ),
    (
        "lambda",
        "lambda",
        OPENING + 'square = lambda value: value * value\n',
        OPENING + 'def square(value):\n    "Multiply a number by itself."\n    return value * value\n',
    ),
    (
        "list-comprehension-inside-list-comprehension",
        "nested-comprehension",
        OPENING + 'rows = [[x for x in row] for row in grid]\n',
        OPENING + 'rows = [x for x in grid]\n',
    ),
    (
        "set-comprehension-inside-set-comprehension",
        "nested-comprehension",
        OPENING + 'sizes = {len({x for x in row}) for row in grid}\n',
        OPENING + 'sizes = {len(row) for row in grid}\n',
    ),
    (
        "dict-comprehension-inside-dict-comprehension",
        "nested-comprehension",
        OPENING + 'table = {key: {inner: 1 for inner in key} for key in keys}\n',
        OPENING + 'table = {key: 1 for key in keys}\n',
    ),
    (
        "generator-expression-inside-generator-expression",
        "nested-comprehension",
        OPENING + 'total = sum(sum(x for x in row) for row in grid)\n',
        OPENING + 'total = sum(len(row) for row in grid)\n',
    ),
    (
        "two-for-clauses",
        "multiple-for-clauses",
        OPENING + 'flat = [x for row in grid for x in row]\n',
        OPENING + 'flat = [x for x in grid]\n',
    ),
    (
        "staticmethod-decorator",
        "decorator",
        OPENING + 'class Tool:\n    "A tool."\n    @staticmethod\n    def run():\n        "Run it."\n        return 1\n',
        OPENING + 'from dataclasses import dataclass\n\n\n@dataclass\nclass Tool:\n    "A tool."\n    size: int = 1\n',
    ),
    (
        "parametrize-decorator",
        "decorator",
        OPENING + 'import pytest\n\n\n@pytest.mark.parametrize("number", [1, 2])\ndef test_number(number):\n    "Check a number."\n    assert number > 0\n',
        OPENING + 'def test_number():\n    "Check numbers in a loop."\n    for number in [1, 2]:\n        assert number > 0\n',
    ),
    (
        "dataclasses-dot-dataclass-decorator",
        "decorator",
        OPENING + 'import dataclasses\n\n\n@dataclasses.dataclass\nclass Tool:\n    "A tool."\n    size: int = 1\n',
        OPENING + 'from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Tool:\n    "A tool."\n    size: int = 1\n',
    ),
    (
        "dataclass-from-another-module",
        "decorator",
        OPENING + 'from other import dataclass\n\n\n@dataclass\nclass Tool:\n    "A tool."\n    size: int = 1\n',
        OPENING + 'from dataclasses import dataclass\n\n\n@dataclass\nclass Tool:\n    "A tool."\n    size: int = 1\n',
    ),
    (
        "dataclass-called-with-positional-argument",
        "decorator",
        OPENING + 'from dataclasses import dataclass\n\n\n@dataclass(True)\nclass Tool:\n    "A tool."\n    size: int = 1\n',
        OPENING + 'from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Tool:\n    "A tool."\n    size: int = 1\n',
    ),
    (
        "star-args-parameter",
        "star-args",
        OPENING + 'def total(*numbers):\n    "Add numbers."\n    return sum(numbers)\n',
        OPENING + 'def total(numbers):\n    "Add a list of numbers."\n    return sum(numbers)\n',
    ),
    (
        "star-kwargs-parameter",
        "star-kwargs",
        OPENING + 'def build(**options):\n    "Build."\n    return options\n',
        OPENING + 'def build(options):\n    "Build."\n    return options\n',
    ),
    (
        "star-unpacking-in-call",
        "call-unpacking",
        OPENING + 'names = ["a", "b"]\nprint(*names)\n',
        OPENING + 'names = ["a", "b"]\nprint(names)\n',
    ),
    (
        "double-star-unpacking-in-call",
        "call-unpacking",
        OPENING + 'options = {"a": 1}\nbuild(**options)\n',
        OPENING + 'options = {"a": 1}\nbuild(options)\n',
    ),
]

# Code that looks close to a rule but is allowed: a bare "*" that makes the
# following parameters keyword-only, and a tuple-unpacking assignment.
ALLOWED_SAMPLES = [
    (
        "bare-star-keyword-only-parameters",
        OPENING + 'def generate(entries, *, seed=None):\n    "Build a result."\n    return entries, seed\n',
    ),
    (
        "dataclass-bare-and-with-keywords",
        OPENING
        + 'from dataclasses import dataclass\n\n\n@dataclass\nclass Plain:\n    "A plain record."\n    size: int = 1\n\n\n'
        + '@dataclass(frozen=True)\nclass Frozen:\n    "A frozen record."\n    size: int = 1\n',
    ),
    (
        "two-comprehensions-side-by-side",
        OPENING + 'first = [x for x in range(3)]\nsecond = {y for y in range(3)}\n',
    ),
]


def write_sample(folder, label, text):
    """Write a sample file into the folder and return its path."""
    path = folder / (label + ".py")
    path.write_text(text, encoding="utf-8")
    return path


def test_each_bad_sample_breaks_only_its_rule(tmp_path):
    """Every bad sample fails with exactly the one rule it was written to break."""
    for label, rule, bad_text, good_text in SAMPLES:
        path = write_sample(tmp_path, "bad-" + label, bad_text)
        failures = find_violations(path)
        assert rule_names(failures) == {rule}, label + ": " + str(failures)


def test_each_matching_good_sample_passes(tmp_path):
    """Every good sample, written to match a bad one, reports no failure at all."""
    for label, rule, bad_text, good_text in SAMPLES:
        path = write_sample(tmp_path, "good-" + label, good_text)
        failures = find_violations(path)
        assert failures == [], label + ": " + str(failures)


def test_allowed_look_alikes_pass(tmp_path):
    """A bare star, a plain dataclass and side-by-side comprehensions are accepted."""
    for label, text in ALLOWED_SAMPLES:
        path = write_sample(tmp_path, "allowed-" + label, text)
        failures = find_violations(path)
        assert failures == [], label + ": " + str(failures)


def test_every_rule_has_a_bad_sample():
    """Each rule the gate enforces appears in at least one bad sample."""
    covered = set()
    for label, rule, bad_text, good_text in SAMPLES:
        covered.add(rule)
    expected = {
        "opening-comment",
        "docstring",
        "lambda",
        "nested-comprehension",
        "multiple-for-clauses",
        "decorator",
        "star-args",
        "star-kwargs",
        "call-unpacking",
    }
    assert covered == expected


def test_file_that_does_not_parse_is_a_failure(tmp_path):
    """A file with a syntax error is reported instead of being skipped."""
    path = write_sample(tmp_path, "broken", OPENING + "def broken(:\n")
    assert rule_names(find_violations(path)) == {"does-not-parse"}


def test_package_python_files_are_found_at_run_time():
    """The discovery finds this very file, so an empty search cannot pass silently."""
    found = sorted(PACKAGE_FOLDER.rglob("*.py"))
    assert Path(__file__).resolve() in found


def test_every_package_python_file_is_readable():
    """Every .py file under the package folder, including this one, passes every rule."""
    problems = []
    for path in sorted(PACKAGE_FOLDER.rglob("*.py")):
        for rule, line in find_violations(path):
            problems.append(str(path.relative_to(PACKAGE_FOLDER)) + ":" + str(line) + " " + rule)
    assert problems == [], "\n".join(problems)
