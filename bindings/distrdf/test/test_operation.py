import unittest

from DistRDF import Operation

import ROOT


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


class HistoModelTests(unittest.TestCase):
    """
    Test that Histo*D operations work only when the histogram model is passed.
    """

    def test_histo1d_with_tuple(self):
        """A tuple can be used as a histogram model."""
        op = Operation.create_op("Histo1D", ("name", "title", 10, 0, 10), "x")
        self.assertIsInstance(op, Operation.Histo)
        self.assertEqual(op.name, "Histo1D")
        self.assertEqual(op.args, [("name", "title", 10, 0, 10), "x"])

    def test_histo1d_with_th1dmodel(self):
        """TH1DModel"""
        op = Operation.create_op("Histo1D", ROOT.RDF.TH1DModel(), "x")
        self.assertIsInstance(op, Operation.Histo)
        self.assertEqual(op.name, "Histo1D")

    def test_histo1d_without_model(self):
        """Creating a histogram without model raises ValueError."""
        with self.assertRaises(ValueError):
            _ = Operation.create_op("Histo1D", "x")

    def test_histo2d_with_th2dmodel(self):
        """TH2DModel"""
        op = Operation.create_op("Histo2D", ROOT.RDF.TH2DModel(), "x", "y")
        self.assertIsInstance(op, Operation.Histo)
        self.assertEqual(op.name, "Histo2D")

    def test_histo2d_without_model(self):
        """Creating a histogram without model raises ValueError."""
        with self.assertRaises(ValueError):
            _ = Operation.create_op("Histo2D", "x", "y")

    def test_histo3d_with_th3dmodel(self):
        """TH3DModel"""
        op = Operation.create_op("Histo3D", ROOT.RDF.TH3DModel(), "x", "y", "z")
        self.assertIsInstance(op, Operation.Histo)
        self.assertEqual(op.name, "Histo3D")

    def test_histo3d_without_model(self):
        """Creating a histogram without model raises ValueError."""
        with self.assertRaises(ValueError):
            _ = Operation.create_op("Histo3D", "x", "y", "z")

    def test_histond_with_thndmodel(self):
        """THnDModel"""
        op = Operation.create_op("HistoND", ROOT.RDF.THnDModel(), ["a", "b", "c", "d"])
        self.assertIsInstance(op, Operation.Histo)
        self.assertEqual(op.name, "HistoND")

    def test_histond_without_model(self):
        """Creating a histogram without model raises ValueError."""
        with self.assertRaises(ValueError):
            _ = Operation.create_op("HistoND", ["a", "b", "c", "d"])
