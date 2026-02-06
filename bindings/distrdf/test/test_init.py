import sys
from importlib import reload
from unittest import TestCase

import DistRDF


class ImportDistRDFTest(TestCase):
    """Test status after importing DistRDF"""

    def test_implicit_import(self):
        """Test that importing DistRDF does not imply initializing the C++ runtime."""
        self.assertFalse("cppyy" in sys.modules)
        reload(DistRDF)
        self.assertFalse("cppyy" in sys.modules)
