import sys

import pytest

from check_backend import *
from check_definepersample import *
from check_friend_trees import *
from check_friend_trees_alignment import *
from check_histo_write import *
from check_include_headers import *
from check_inv_mass import *
from check_reducer_merge import *
from check_rungraphs import *
from check_variations import *

if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
