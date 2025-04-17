import sys
from importlib import reload
from unittest import TestCase

import DistRDF


class ImportDistRDFTest(TestCase):
    """Test status after importing DistRDF"""

    def test_implicit_import(self):
        """Test that importing DistRDF does not implicitly import ROOT."""
        self.assertFalse("ROOT" in sys.modules)
        reload(DistRDF)
        self.assertFalse("ROOT" in sys.modules)
