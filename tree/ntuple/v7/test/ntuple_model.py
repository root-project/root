import unittest

import ROOT

RNTupleModel = ROOT.Experimental.RNTupleModel
RFieldZero = ROOT.Experimental.RFieldZero


class NTupleModel(unittest.TestCase):
    """Various tests for the RNTupleModel class"""

    def test_create_model(self):
        """A model can be created."""

        model = RNTupleModel.Create()
        self.assertTrue(model)

    def test_create_bare_model(self):
        """A bare model can be created."""

        model = RNTupleModel.CreateBare()
        self.assertTrue(model)


if __name__ == "__main__":
    unittest.main()
