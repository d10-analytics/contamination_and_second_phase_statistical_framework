# This file turns the user's description of particle classes (the "entries")
# into a checked list of plain dictionaries. An entry says which class a
# particle belongs to, which distribution its diameters follow, how many
# particles to keep and the distribution's parameters. The text can be one JSON
# array, one JSON object, or JSONL (one JSON object per line). Every check lives
# in one function, validate_entries, so text input, file input and a Python list
# are all judged by exactly the same rules. An ordinary problem with the input
# (a wrong key, a bad value, broken JSON) is reported as a ValueError whose
# message names the entry number (counting from 1) or the line. Extreme inputs
# that Python itself refuses, such as an integer of thousands of digits or
# arrays nested a hundred thousand levels deep, raise Python's own error.

import json
import math
import unicodedata
from pathlib import Path

# The keys every entry has, whatever its distribution.
COMMON_KEYS = ("class", "distribution", "N")

# The parameter keys of each distribution, in the order they are written out.
# "scale" is the spread of the distribution, not a rate (1 / scale).
PARAMETER_KEYS = {
    "gaussian": ("mean", "stdev"),
    "gamma": ("shape", "scale"),
    "gev": ("shape", "loc", "scale"),
}

# Parameters that must be strictly above zero for each distribution. A gev
# shape may be any finite number, so it is not listed.
POSITIVE_KEYS = {
    "gaussian": ("stdev",),
    "gamma": ("shape", "scale"),
    "gev": ("scale",),
}

# These two characters are not control characters, yet str.splitlines treats
# them as line breaks, so a class name may not contain them.
LINE_BREAK_CHARACTERS = ("\u2028", "\u2029")

# The characters a JSON value can start with. Text after the first complete
# value is only treated as more JSONL when it starts like this; anything else
# (a stray bracket, a comma) is an ordinary JSON error in the whole text.
VALUE_START_CHARACTERS = '{["-0123456789tfnNI'

READ_ENTRIES_HINT = (
    " If this text is the path of a file, call read_entries(path) instead of parse_entries."
)


def describe_class_problem(name):
    """Return a short reason why a class name is not allowed, or None when it is fine."""
    if not isinstance(name, str):
        return "'class' must be a string"
    if name == "":
        return "'class' must not be empty"
    if name != name.strip():
        return "'class' must not start or end with whitespace"
    for character in name:
        category = unicodedata.category(character)
        if category == "Cc":
            return "'class' must not contain a control character"
        if category == "Cs":
            return "'class' must not contain a lone surrogate"
        if character in LINE_BREAK_CHARACTERS:
            return "'class' must not contain a line or paragraph separator"
    return None


def describe_count_problem(count):
    """Return a short reason why N is not allowed, or None when it is fine."""
    # Booleans are checked first because Python treats True as the integer 1.
    if isinstance(count, bool) or not isinstance(count, int):
        return "'N' must be a whole number written as a JSON integer, not " + repr(count)
    if count < 1:
        return "'N' must be at least 1, not " + repr(count)
    return None


def convert_parameter(key, value, label):
    """Return a parameter as a finite float, or raise ValueError naming the entry."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: '{key}' must be a number, not {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{label}: '{key}' is too large to use as a number")
    if not math.isfinite(number):
        raise ValueError(f"{label}: '{key}' must be a finite number, not {value!r}")
    return number


def validate_entry(entry, label):
    """Check one entry dictionary and return a clean copy, naming it by label in any error."""
    if not isinstance(entry, dict):
        raise ValueError(f"{label}: an entry must be a JSON object, not {entry!r}")
    for key in COMMON_KEYS:
        if key not in entry:
            raise ValueError(f"{label}: missing key '{key}'")
    distribution = entry["distribution"]
    if not isinstance(distribution, str) or distribution not in PARAMETER_KEYS:
        raise ValueError(
            f"{label}: 'distribution' must be exactly gaussian, gamma or gev, not {distribution!r}"
        )
    parameter_keys = PARAMETER_KEYS[distribution]
    for key in parameter_keys:
        if key not in entry:
            raise ValueError(f"{label}: missing key '{key}' for a {distribution} entry")
    allowed_keys = COMMON_KEYS + parameter_keys
    for key in entry:
        if key not in allowed_keys:
            raise ValueError(f"{label}: unexpected key {key!r} for a {distribution} entry")
    class_problem = describe_class_problem(entry["class"])
    if class_problem is not None:
        raise ValueError(f"{label}: {class_problem}, got {entry['class']!r}")
    count_problem = describe_count_problem(entry["N"])
    if count_problem is not None:
        raise ValueError(f"{label}: {count_problem}")
    clean = {
        "class": entry["class"],
        "distribution": distribution,
        "N": entry["N"],
    }
    for key in parameter_keys:
        clean[key] = convert_parameter(key, entry[key], label)
    for key in POSITIVE_KEYS[distribution]:
        if clean[key] <= 0:
            raise ValueError(f"{label}: '{key}' must be above 0, not {clean[key]!r}")
    return clean


def validate_entries(entries):
    """Check a list of entry dictionaries and return a new list of clean dictionaries."""
    if not isinstance(entries, list):
        raise ValueError("entries must be a list of entry dictionaries")
    if len(entries) == 0:
        raise ValueError("entries must not be empty: give at least one entry")
    clean_entries = []
    for position, entry in enumerate(entries, start=1):
        clean_entries.append(validate_entry(entry, f"entry {position}"))
    return clean_entries


def skip_whitespace(text):
    """Return the position of the first character in text that is not JSON whitespace."""
    position = 0
    while position < len(text) and text[position] in " \t\r\n":
        position += 1
    return position


def describe_json_error(error, hint):
    """Return the message for text that is not valid JSON, with JSON's own line and column."""
    return f"invalid JSON: {error.msg} (line {error.lineno}, column {error.colno})." + hint


def parse_lines(text, hint):
    """Parse JSONL text (one JSON object per line, blank lines skipped) into entry dictionaries."""
    clean_entries = []
    # Splitting on the newline character only keeps line numbers equal to what
    # an editor shows, even if a stray separator character is inside a value.
    for line_number, line in enumerate(text.split("\n"), start=1):
        if line.strip(" \t\r") == "":
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSONL line {line_number}: invalid JSON: {error.msg} (column {error.colno})." + hint
            )
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number}: each line must be a JSON object")
        clean_entries.append(validate_entry(value, f"JSONL line {line_number}"))
    return clean_entries


def parse_text(text, hint):
    """Parse entry text; the hint is added to messages about text that is not JSON."""
    if not isinstance(text, str):
        raise ValueError("entry text must be a string; to read a file call read_entries(path)")
    start = skip_whitespace(text)
    if start == len(text):
        raise ValueError("entry text is empty: give a JSON array, a JSON object or JSONL")
    try:
        value, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as error:
        raise ValueError(describe_json_error(error, hint))
    rest = text[end:]
    rest_start = skip_whitespace(rest)
    if rest_start < len(rest):
        if rest[rest_start] in VALUE_START_CHARACTERS:
            # Another value follows the first, so this is JSONL.
            return parse_lines(text, hint)
        # Anything else is a plain JSON mistake; json.loads reports it with
        # the line and column in the whole text.
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(describe_json_error(error, hint))
    if isinstance(value, dict):
        return validate_entries([value])
    if isinstance(value, list):
        return validate_entries(value)
    raise ValueError(
        f"entry text must be a JSON array of objects or one JSON object, not {value!r}." + hint
    )


def parse_entries(text):
    """Turn JSON array, single-object JSON or JSONL text into a checked list of entry dictionaries."""
    return parse_text(text, READ_ENTRIES_HINT)


def read_entries(path):
    """Read a UTF-8 JSON or JSONL file and return the same checked list that parse_entries gives."""
    file_path = Path(path)
    if file_path.is_dir():
        raise ValueError(f"cannot read entries: '{path}' is a directory, not a file")
    if not file_path.exists():
        raise ValueError(f"cannot read entries: '{path}' does not exist")
    try:
        text = file_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"cannot read entries from '{path}' as UTF-8 text: {error}")
    try:
        return parse_text(text, "")
    except ValueError as error:
        raise ValueError(f"{path}: {error}")
