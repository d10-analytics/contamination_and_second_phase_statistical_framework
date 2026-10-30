# This folder is the synthetic particle measurement generator. The four names
# imported below are its public interface: generate_dataset builds a dataset in
# memory, write_outputs saves it as a CSV plus a summary, and parse_entries and
# read_entries turn JSON or JSONL text or a file into checked entries.

from synthetic_data_generation.dataset import generate_dataset
from synthetic_data_generation.entries import parse_entries, read_entries
from synthetic_data_generation.outputs import write_outputs
