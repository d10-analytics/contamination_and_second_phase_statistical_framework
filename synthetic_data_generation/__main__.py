# This file lets the generator run with "python -m synthetic_data_generation".
# It hands the command line to cli.main and exits with the code that returns.

import sys

from synthetic_data_generation import cli

sys.exit(cli.main(sys.argv[1:]))
