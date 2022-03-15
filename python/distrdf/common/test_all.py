import sys

import pytest

from check_rungraphs import *

if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
