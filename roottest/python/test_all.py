#! /usr/bin/env python
"""
Test runner interface
---------------------

This is the pytest (see http://pytest.org) based test runner taken from
PyPy. This way, PyROOT and PyPy/cppyy can share the same set of tests.

For more information, use test_all.py -h.
"""

import os, sys


if __name__ == '__main__':
    if len(sys.argv) == 1 and os.path.dirname(sys.argv[0]) in '.':
        print >> sys.stderr, __doc__
        sys.exit(2)
    # add local pytest dir to sys.path
    toplevel = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, os.path.join(toplevel, 'pytest'))
    import pytest
    import pytest_cov
    sys.exit(pytest.main(plugins=[pytest_cov]))
