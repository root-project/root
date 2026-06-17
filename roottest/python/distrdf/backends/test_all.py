import shlex
import sys

import pytest

# Avoid errors from linters like "unable to detect undefined names"
from check_backend import *  # noqa: F403
from check_cloned_actions import *  # noqa: F403
from check_definepersample import *  # noqa: F403
from check_distribute_cppcode import *  # noqa: F403
from check_distribute_headers_sharedlibs_files import *  # noqa: F403
from check_explicit_api import *  # noqa: F403
from check_friend_trees import *  # noqa: F403
from check_friend_trees_alignment import *  # noqa: F403
from check_fromspec import *  # noqa: F403
from check_histo_write import *  # noqa: F403
from check_inv_mass import *  # noqa: F403
from check_live_visualize import *  # noqa: F403
from check_missing_values import *  # noqa: F403
from check_reducer_merge import *  # noqa: F403
from check_rungraphs import *  # noqa: F403
from check_variations import *  # noqa: F403

if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    sys.exit(pytest.main(args=shlex.split(f"{__file__} -x -vvv")))
