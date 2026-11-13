# This folder is the synthetic particle measurement generator. The six names
# imported below are its public interface: generate_dataset builds a dataset in
# memory, write_outputs saves it as a CSV plus a summary, parse_entries and
# read_entries turn JSON or JSONL text or a file into checked entries, and
# parse_measurements and read_measurements turn the CSV text or file back into
# one dictionary per particle.

from synthetic_data_generation.dataset import generate_dataset
from synthetic_data_generation.entries import parse_entries, read_entries
from synthetic_data_generation.outputs import parse_measurements, read_measurements, write_outputs
