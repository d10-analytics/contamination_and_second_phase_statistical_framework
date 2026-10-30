# This file checks the three small repository files that support the synthetic
# data package: the list of required packages, the pointer section in the
# top-level README, and the ignore list. It reads each file as plain text and
# compares the exact lines, so changing or deleting any promised line makes a
# named test fail.

from pathlib import Path

# The package folder and the repository root are found from this file's own
# location, so the tests work from any starting folder.
PACKAGE_FOLDER = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = PACKAGE_FOLDER.parent

EXPECTED_REQUIREMENTS = ["numpy>=2.2,<3", "scipy>=1.15,<2", "matplotlib>=3.9,<4"]
GUIDE_LINK = "](synthetic_data_generation/README.md)"
OUTPUT_IGNORE_LINE = "synthetic_data_generation/example/output/"


def read_lines(path):
    """Return the lines of a text file, without line endings."""
    return path.read_text(encoding="utf-8").splitlines()


def test_requirements_open_with_a_comment():
    """The requirements file must start with a "#" comment explaining itself."""
    lines = read_lines(PACKAGE_FOLDER / "requirements.txt")
    assert lines[0].startswith("#")


def test_requirements_list_exactly_the_three_bounded_packages():
    """Apart from comments and blank lines, only the three bounded packages are listed."""
    lines = read_lines(PACKAGE_FOLDER / "requirements.txt")
    requirement_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        requirement_lines.append(stripped)
    assert requirement_lines == EXPECTED_REQUIREMENTS


def test_root_readme_links_to_the_package_guide():
    """The top-level README must link to the package guide by its exact path."""
    text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert GUIDE_LINK in text


def test_ignore_file_keeps_agents_line_and_adds_output_folder():
    """The ignore list keeps its AGENTS.md line and ignores the example output folder."""
    lines = read_lines(REPOSITORY_ROOT / ".gitignore")
    assert lines == ["AGENTS.md", OUTPUT_IGNORE_LINE]
