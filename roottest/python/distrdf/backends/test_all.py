import sys

import pytest

from check_backend import *
from check_cloned_actions import *
from check_distribute_cppcode import * 
from check_definepersample import *
from check_fromspec import *
from check_explicit_api import *
from check_friend_trees_alignment import *
from check_friend_trees import *
from check_histo_write import *
from check_distribute_headers_sharedlibs_files import *
from check_inv_mass import *
from check_live_visualize import *
from check_missing_values import *
from check_reducer_merge import *
from check_rungraphs import *
from check_variations import *

if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=[__file__]))
