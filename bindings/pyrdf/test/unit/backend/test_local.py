import unittest
import ROOT
import PyRDF
from PyRDF.backend.Local import Local


class SelectionTest(unittest.TestCase):
    """Check the accuracy of 'PyRDF.use' method."""

    def test_local_select(self):
        """Check if 'local' environment gets set correctly."""
        PyRDF.use("local")
        self.assertIsInstance(PyRDF.current_backend, Local)


class OperationSupportTest(unittest.TestCase):
    """
    Ensure that incoming operations are classified accurately in local
    environments.

    """

    def test_action(self):
        """Check that action nodes are classified accurately."""
        backend = Local()
        backend.check_supported("Count")

    def test_transformation(self):
        """Check that transformation nodes are classified accurately."""
        backend = Local()
        backend.check_supported("Define")

    def test_unsupported_operations(self):
        """Check that unsupported operations raise an Exception."""
        backend = Local()
        with self.assertRaises(Exception):
            backend.check_supported("Take")

        with self.assertRaises(Exception):
            backend.check_supported("Foreach")

    def test_none(self):
        """Check that incorrect operations raise an Exception."""
        backend = Local()
        with self.assertRaises(Exception):
            backend.check_supported("random")

    def test_range_operation_single_thread(self):
        """
        Check that 'Range' operation works in single-threaded mode and raises an
        Exception in multi-threaded mode.

        """
        backend = Local()
        backend.check_supported("Range")

    def test_range_operation_multi_thread(self):
        """
        Check that 'Range' operation raises an Exception in multi-threaded
        mode.

        """
        ROOT.ROOT.EnableImplicitMT()
        backend = Local()
        with self.assertRaises(Exception):
            backend.check_supported("Range")

        ROOT.ROOT.DisableImplicitMT()


class InitializationTest(unittest.TestCase):
    """Check initialization method in the Local backend"""

    def test_initialization_method(self):
        """
        Check initialization method in Local backend.

        Define a method in the ROOT interpreter called getValue which returns
        the value defined by the user on the python side.

        """
        def init(value):
            cpp_code = '''auto getUserValue = [](){return %s ;};''' % value
            ROOT.gInterpreter.Declare(cpp_code)

        PyRDF.initialize(init, 123)
        PyRDF.current_backend = Local()
        df = PyRDF.RDataFrame(1)
        s = df.Define("userValue", "getUserValue()").Sum("userValue")
        self.assertEqual(s.GetValue(), 123)
