# This file checks the rules for the placement visualization folder that no
# other test covers: no file in it imports OpenCV, pandas or matplotlib's
# pyplot interface, and it has no __main__.py, so there is no command line. It
# reads every .py file as a syntax tree (the shape Python itself builds from
# the code), and proves the scan on small throwaway sources so that removing
# a check makes a test fail.

import ast
from pathlib import Path

VISUALIZATION_FOLDER = Path(__file__).resolve().parent.parent
BANNED_MODULES = ("cv2", "pandas", "matplotlib.pyplot")


def imported_modules(tree):
    """Return every module name a file imports, counting "from matplotlib import pyplot" as matplotlib.pyplot."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
            for alias in node.names:
                names.append(node.module + "." + alias.name)
    return names


def banned_imports(source):
    """Return each banned module the source imports, once, by exact name or as a submodule."""
    found = []
    for name in imported_modules(ast.parse(source)):
        for banned in BANNED_MODULES:
            if name == banned or name.startswith(banned + "."):
                found.append(banned)
    return sorted(set(found))


def test_the_scan_finds_each_banned_import_form():
    """Every way of importing a banned module is reported, and ordinary matplotlib imports are not."""
    assert banned_imports("import cv2") == ["cv2"]
    assert banned_imports("import pandas as pd") == ["pandas"]
    assert banned_imports("import matplotlib.pyplot as plt") == ["matplotlib.pyplot"]
    assert banned_imports("from matplotlib import pyplot") == ["matplotlib.pyplot"]
    assert banned_imports("from matplotlib.pyplot import figure") == ["matplotlib.pyplot"]
    assert banned_imports("from cv2 import imread") == ["cv2"]
    assert banned_imports("from matplotlib.figure import Figure\nimport numpy") == []


def test_no_file_in_the_folder_imports_a_banned_module():
    """Every .py file in the folder, tests included, is free of OpenCV, pandas and pyplot imports."""
    files = sorted(VISUALIZATION_FOLDER.rglob("*.py"))
    assert Path(__file__).resolve() in files
    problems = []
    for path in files:
        for banned in banned_imports(path.read_text(encoding="utf-8")):
            problems.append(f"{path.name} imports {banned}")
    assert problems == []


def test_the_folder_has_no_main_module():
    """There is no __main__.py anywhere in the folder, so it cannot be run as a command line."""
    assert list(VISUALIZATION_FOLDER.rglob("__main__.py")) == []


def test_the_package_exports_the_still_image_writer():
    """The folder is importable and exports write_still_images."""
    import placement_visualization

    assert callable(placement_visualization.write_still_images)
