from DistRDF.Operation import Operation
import unittest


class ClassifyTest(unittest.TestCase):
    """Ensure that incoming operations are classified accurately."""

    def test_action(self):
        """Action nodes are classified accurately."""
        op = Operation("Count")
        self.assertEqual(op.op_type, Operation.ACTION)

    def test_instant_action(self):
        """Instant actions are classified accurately."""
        op = Operation("Snapshot")
        self.assertEqual(op.op_type, Operation.INSTANT_ACTION)

    def test_transformation(self):
        """Transformation nodes are classified accurately."""
        op = Operation("Define", "c1")
        self.assertEqual(op.op_type, Operation.TRANSFORMATION)

    def test_none(self):
        """Incorrect operations raise an Exception."""
        with self.assertRaises(Exception):
            Operation("random")


class ArgsTest(unittest.TestCase):
    """Ensure that arguments and named arguments are set accurately."""

    def test_without_kwargs(self):
        """Check that unnamed arguments are properly set."""
        op = Operation("Define", 1, "b")
        self.assertEqual(op.args, [1, "b"])
        self.assertEqual(op.kwargs, {})

    def test_without_args(self):
        """Check that no unnamed arguments are properly set."""
        op = Operation("Define", a=1, b="b")
        self.assertEqual(op.args, [])
        self.assertEqual(op.kwargs, {"a": 1, "b": "b"})

    def test_with_args_and_kwargs(self):
        """Check that named and unnamed arguments are properly set."""
        op = Operation("Define", 2, "p", a=1, b="b")
        self.assertEqual(op.args, [2, "p"])
        self.assertEqual(op.kwargs, {"a": 1, "b": "b"})

    def test_without_args_and_kwargs(self):
        """Check Operation constructor without arguments."""
        op = Operation("Define")
        self.assertEqual(op.args, [])
        self.assertEqual(op.kwargs, {})
