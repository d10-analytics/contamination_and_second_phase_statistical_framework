# This file tests how the generator reads and checks particle entries. Each test
# feeds real text or a real file to parse_entries, read_entries or the shared
# list checker validate_entries, and asserts the exact result or the specific
# message of the error. Bad inputs are tried in loops of (input, expected
# message part) pairs, so a failure shows which pair broke.

import json

import pytest

from synthetic_data_generation import entries

GAUSSIAN = {"class": "A", "distribution": "gaussian", "N": 3, "mean": 10.0, "stdev": 2.0}
GAMMA = {"class": "A", "distribution": "gamma", "N": 4, "shape": 2.0, "scale": 3.0}
GEV = {"class": "B", "distribution": "gev", "N": 5, "shape": 0.5, "loc": 1.0, "scale": 2.0}


def with_change(base, key, value):
    """Return a copy of an entry dictionary with one key set or replaced."""
    copy = dict(base)
    copy[key] = value
    return copy


def without_key(base, key):
    """Return a copy of an entry dictionary with one key removed."""
    copy = dict(base)
    del copy[key]
    return copy


def error_message(call, argument):
    """Call a function with one argument, expect a ValueError and return its message."""
    with pytest.raises(ValueError) as caught:
        call(argument)
    return str(caught.value)


def test_array_object_and_jsonl_give_equal_lists():
    """The same entries written three ways parse to equal plain dictionaries in input order."""
    expected = [GAUSSIAN, GAMMA, GEV, with_change(GAUSSIAN, "N", 7)]
    array_text = json.dumps(expected, indent=2)
    jsonl_text = "\n\n".join(json.dumps(item) for item in expected) + "\n\n"
    assert entries.parse_entries(array_text) == expected
    assert entries.parse_entries(jsonl_text) == expected
    assert entries.parse_entries(json.dumps(GEV)) == [GEV]
    assert entries.parse_entries(json.dumps(GEV, indent=4)) == [GEV]
    for item in entries.parse_entries(array_text):
        assert type(item) is dict
    assert list(entries.parse_entries(array_text)[0]) == [
        "class", "distribution", "N", "mean", "stdev"
    ]


def test_class_names_are_kept_verbatim_and_case_sensitive():
    """Commas, quotes and non-ASCII letters survive, and IN738 and in738 stay separate."""
    names = ['a,b', 'say "hi"', "Nickel-Niéß中", "IN738", "in738", "IN738"]
    items = []
    for name in names:
        items.append(with_change(GAUSSIAN, "class", name))
    parsed = entries.parse_entries(json.dumps(items))
    assert [item["class"] for item in parsed] == names
    assert len(parsed) == 6


def test_parameters_are_returned_as_floats_and_count_stays_integer():
    """Integer parameters become floats while N remains an integer."""
    text = '{"class":"A","distribution":"gaussian","N":3,"mean":10,"stdev":2}'
    item = entries.parse_entries(text)[0]
    assert type(item["mean"]) is float
    assert type(item["N"]) is int
    assert item["mean"] == 10.0


def test_gev_shape_may_be_negative_zero_or_positive():
    """A gev shape of -0.5, 0 and 0.5 is accepted."""
    for shape in (-0.5, 0, 0.5):
        parsed = entries.parse_entries(json.dumps([with_change(GEV, "shape", shape)]))
        assert parsed[0]["shape"] == float(shape)


def test_third_jsonl_line_error_names_line_three_after_a_blank_line():
    """An error on the third physical line says line 3 even though a blank line was skipped."""
    text = json.dumps(GAUSSIAN) + "\n\n" + '{"class": "A", oops}\n'
    assert "JSONL line 3" in error_message(entries.parse_entries, text)
    bad_entry = json.dumps(GAUSSIAN) + "\n\n" + json.dumps(with_change(GAMMA, "scale", 0))
    message = error_message(entries.parse_entries, bad_entry)
    assert "JSONL line 3" in message
    assert "'scale' must be above 0" in message
    non_object = json.dumps(GAUSSIAN) + "\n[1]\n"
    assert "JSONL line 2: each line must be a JSON object" in error_message(
        entries.parse_entries, non_object
    )


def test_broken_array_reports_json_position_not_a_jsonl_line():
    """A broken multi-line array keeps JSON's line and column and is never retried as JSONL."""
    text = '[\n  {"class": "A",\n   "distribution": "gaussian",\n   oops\n  }\n]'
    message = error_message(entries.parse_entries, text)
    assert "invalid JSON" in message
    assert "line 4, column 4" in message
    assert "JSONL" not in message


def test_text_after_a_complete_array_reports_json_position_not_a_jsonl_line():
    """A stray bracket, comma or word after an indented array keeps JSON's own line and column."""
    lines = []
    for number in range(1, 14):
        lines.append('  {"class": "C%d", "distribution": "gaussian", "N": 1, "mean": 1, "stdev": 1},' % number)
    lines.append('  {"class": "C14", "distribution": "gaussian", "N": 1, "mean": 1, "stdev": 1}')
    array_text = "[\n" + "\n".join(lines) + "\n]"
    assert len(array_text.split("\n")) == 16
    cases = [
        (array_text + "\n]\n", "line 17, column 1"),
        (array_text + ",\n", "line 16, column 2"),
        (array_text + "\nx\n", "line 17, column 1"),
    ]
    for text, position in cases:
        message = error_message(entries.parse_entries, text)
        assert "invalid JSON" in message, text[-6:]
        assert position in message, (text[-6:], message)
        assert "JSONL" not in message, text[-6:]


def test_top_level_value_empty_and_non_json_text_are_rejected():
    """Scalars, arrays of non-objects, empty arrays and empty text each get their own message."""
    cases = [
        ("5", "must be a JSON array of objects or one JSON object"),
        ('"text"', "must be a JSON array of objects or one JSON object"),
        ("null", "must be a JSON array of objects or one JSON object"),
        ("[1, 2]", "entry 1: an entry must be a JSON object"),
        ('[{"class": "A"}, 3]', "entry 1: missing key 'distribution'"),
        ("[]", "must not be empty"),
        ("", "entry text is empty"),
        ("  \n\t ", "entry text is empty"),
    ]
    for text, expected in cases:
        assert expected in error_message(entries.parse_entries, text), text


def test_second_entry_that_is_not_an_object_names_entry_two():
    """A non-object inside an array is named by its entry number."""
    text = json.dumps([GAUSSIAN, 3])
    assert "entry 2: an entry must be a JSON object" in error_message(entries.parse_entries, text)


def test_text_that_is_not_json_names_read_entries():
    """A file name passed as text produces a message that points to read_entries."""
    message = error_message(entries.parse_entries, "example_input.jsonl")
    assert "read_entries" in message
    assert "invalid JSON" in message


def test_distribution_must_be_exact():
    """Only gaussian, gamma and gev in lower case are accepted, and the entry is named."""
    for name in ("lognormal", "Gaussian", "", 3, None):
        text = json.dumps([GAUSSIAN, with_change(GAMMA, "distribution", name)])
        message = error_message(entries.parse_entries, text)
        assert message.startswith("entry 2: 'distribution' must be exactly gaussian, gamma or gev")


def test_missing_and_extra_keys_are_rejected():
    """Each missing key and each extra key is named, with the entry number."""
    for base in (GAUSSIAN, GAMMA, GEV):
        for key in base:
            text = json.dumps([GAUSSIAN, without_key(base, key)])
            assert f"entry 2: missing key '{key}'" in error_message(entries.parse_entries, text)
    extra_cases = [
        (with_change(GAMMA, "rate", 1.0), "rate"),
        (with_change(GAUSSIAN, "loc", 1.0), "loc"),
        (with_change(GEV, "mean", 1.0), "mean"),
        (with_change(GAUSSIAN, "colour", "red"), "colour"),
    ]
    for item, key in extra_cases:
        message = error_message(entries.parse_entries, json.dumps([GAMMA, item]))
        assert f"entry 2: unexpected key '{key}'" in message


def test_count_must_be_a_positive_json_integer():
    """N of 0, -1, 3.0, true and "3" is rejected with the entry number."""
    cases = [
        ("0", "'N' must be at least 1, not 0"),
        ("-1", "'N' must be at least 1, not -1"),
        ("3.0", "'N' must be a whole number written as a JSON integer, not 3.0"),
        ("true", "'N' must be a whole number written as a JSON integer, not True"),
        ('"3"', "'N' must be a whole number written as a JSON integer, not '3'"),
    ]
    for literal, expected in cases:
        body = '{"class":"A","distribution":"gaussian","N":%s,"mean":1,"stdev":1}' % literal
        message = error_message(entries.parse_entries, "[" + body + "]")
        assert message.startswith("entry 1: "), literal
        assert expected in message, literal


def test_parameters_must_be_finite_real_numbers():
    """NaN, infinities, booleans, strings and a 400-digit integer name the entry and key."""
    cases = [
        ("NaN", "must be a finite number"),
        ("Infinity", "must be a finite number"),
        ("-Infinity", "must be a finite number"),
        ("true", "must be a number, not True"),
        ("false", "must be a number, not False"),
        ('"1.5"', "must be a number, not '1.5'"),
        ("1" + "0" * 399, "is too large to use as a number"),
        ("1e999", "must be a finite number"),
        ("null", "must be a number, not None"),
    ]
    for literal, expected in cases:
        body = '{"class":"A","distribution":"gaussian","N":2,"mean":%s,"stdev":1}' % literal
        message = error_message(entries.parse_entries, "[" + body + "]")
        assert message.startswith("entry 1: 'mean' "), literal
        assert expected in message, literal


def test_large_integer_in_a_list_is_rejected_without_overflow_error():
    """The shared list checker turns a too-large integer into the named entry message."""
    huge = with_change(GAMMA, "scale", 10**400)
    assert "entry 2: 'scale' is too large" in error_message(entries.validate_entries, [GAMMA, huge])


def test_stdev_scale_and_gamma_shape_must_be_above_zero():
    """Zero and negative values are rejected for stdev, every scale and the gamma shape."""
    cases = [
        (GAUSSIAN, "stdev"),
        (GAMMA, "scale"),
        (GAMMA, "shape"),
        (GEV, "scale"),
    ]
    for base, key in cases:
        for bad in (0, -1.5):
            message = error_message(entries.parse_entries, json.dumps([with_change(base, key, bad)]))
            assert message.startswith(f"entry 1: '{key}' must be above 0")


def test_class_must_be_a_clean_non_empty_string():
    """Non-strings, empty, padded, control, separator and lone-surrogate names are rejected."""
    cases = [
        (5, "'class' must be a string"),
        (None, "'class' must be a string"),
        ("", "'class' must not be empty"),
        (" A", "must not start or end with whitespace"),
        ("A ", "must not start or end with whitespace"),
        ("A\tB", "must not contain a control character"),
        ("A\x00B", "must not contain a control character"),
        ("A\u2028B", "must not contain a line or paragraph separator"),
        ("A\u2029B", "must not contain a line or paragraph separator"),
        ("A\ud800B", "must not contain a lone surrogate"),
    ]
    for name, expected in cases:
        text = json.dumps([GAUSSIAN, with_change(GAUSSIAN, "class", name)])
        message = error_message(entries.parse_entries, text)
        assert message.startswith("entry 2: "), repr(name)
        assert expected in message, repr(name)


def test_raw_json_text_with_escaped_surrogate_and_tab_is_rejected():
    """Escapes written in the JSON text itself reach the same class check."""
    surrogate = '[{"class":"\\ud800","distribution":"gaussian","N":1,"mean":1,"stdev":1}]'
    assert "lone surrogate" in error_message(entries.parse_entries, surrogate)
    separator = '[{"class":"a\\u2028b","distribution":"gaussian","N":1,"mean":1,"stdev":1}]'
    assert "separator" in error_message(entries.parse_entries, separator)


def test_validate_entries_rejects_non_lists_and_returns_copies():
    """The list checker needs a non-empty list and never changes the caller's dictionaries."""
    assert "must be a list" in error_message(entries.validate_entries, GAUSSIAN)
    assert "must not be empty" in error_message(entries.validate_entries, [])
    original = dict(GAUSSIAN)
    result = entries.validate_entries([original])
    assert result == [GAUSSIAN]
    assert result[0] is not original


def test_read_entries_matches_parse_entries_for_jsonl_and_array_files(tmp_path):
    """Reading a UTF-8 file gives exactly what parsing its text gives, including non-ASCII."""
    items = [with_change(GAUSSIAN, "class", "Ni\u00e9\u4e2d"), GAMMA, GEV]
    array_file = tmp_path / "entries.json"
    array_file.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    jsonl_file = tmp_path / "entries.jsonl"
    lines = []
    for item in items:
        lines.append(json.dumps(item, ensure_ascii=False))
    jsonl_file.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    for path in (array_file, jsonl_file):
        text = path.read_text(encoding="utf-8")
        assert entries.read_entries(path) == entries.parse_entries(text)
        assert entries.read_entries(str(path)) == entries.parse_entries(text)
        assert entries.read_entries(path)[0]["class"] == "Nié中"


def test_read_entries_reports_missing_path_and_directory(tmp_path):
    """A missing path and a directory each raise an error that contains the path."""
    missing = tmp_path / "nothing.jsonl"
    assert str(missing) in error_message(entries.read_entries, missing)
    assert str(tmp_path) in error_message(entries.read_entries, tmp_path)


def test_read_entries_reports_entry_number_or_line_for_bad_content(tmp_path):
    """An invalid entry inside a file is named by entry number or JSONL line."""
    array_file = tmp_path / "bad.json"
    array_file.write_text(json.dumps([GAUSSIAN, with_change(GAMMA, "N", 0)]), encoding="utf-8")
    message = error_message(entries.read_entries, array_file)
    assert "entry 2: 'N' must be at least 1" in message
    assert str(array_file) in message
    jsonl_file = tmp_path / "bad.jsonl"
    jsonl_file.write_text(json.dumps(GAUSSIAN) + "\n\n{broken\n", encoding="utf-8")
    assert "JSONL line 3" in error_message(entries.read_entries, jsonl_file)
    empty_file = tmp_path / "empty.json"
    empty_file.write_text("", encoding="utf-8")
    assert "entry text is empty" in error_message(entries.read_entries, empty_file)


def test_read_entries_rejects_a_file_that_is_not_utf8(tmp_path):
    """Bytes that are not UTF-8 give a message naming the path."""
    bad_file = tmp_path / "latin.jsonl"
    bad_file.write_bytes(b'{"class": "caf\xe9"}')
    message = error_message(entries.read_entries, bad_file)
    assert str(bad_file) in message
    assert "UTF-8" in message
