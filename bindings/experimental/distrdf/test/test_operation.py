from DistRDF import Operation
import unittest


class ClassifyTest(unittest.TestCase):
    """Ensure that incoming operations are classified accurately."""

    def test_action(self):
        """Action nodes are classified accurately."""
        op = Operation.create_op("Count")
        self.assertIsInstance(op, Operation.Action)

    def test_instant_action(self):
        """Instant actions are classified accurately."""
        op = Operation.create_op("Snapshot")
        self.assertIsInstance(op, Operation.InstantAction)

    def test_transformation(self):
        """Transformation nodes are classified accurately."""
        op = Operation.create_op("Define")
        self.assertIsInstance(op, Operation.Transformation)

    def test_none(self):
        """Incorrect operations raise an Exception."""
        with self.assertRaises(ValueError):
            Operation.create_op("random")


class ArgsTest(unittest.TestCase):
    """Ensure that arguments and named arguments are set accurately."""

    def test_without_kwargs(self):
        """Check that unnamed arguments are properly set."""
        op = Operation.create_op("Define", 1, "b")
        self.assertEqual(op.args, [1, "b"])
        self.assertEqual(op.kwargs, {})

    def test_without_args(self):
        """Check that no unnamed arguments are properly set."""
        op = Operation.create_op("Define", a=1, b="b")
        self.assertEqual(op.args, [])
        self.assertEqual(op.kwargs, {"a": 1, "b": "b"})

    def test_with_args_and_kwargs(self):
        """Check that named and unnamed arguments are properly set."""
        op = Operation.create_op("Define", 2, "p", a=1, b="b")
        self.assertEqual(op.args, [2, "p"])
        self.assertEqual(op.kwargs, {"a": 1, "b": "b"})

    def test_without_args_and_kwargs(self):
        """Check Operation constructor without arguments."""
        op = Operation.create_op("Define")
        self.assertEqual(op.args, [])
        self.assertEqual(op.kwargs, {})
