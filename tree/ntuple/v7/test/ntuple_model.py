import unittest

import ROOT

RNTupleModel = ROOT.Experimental.RNTupleModel
RNTupleWriteOptions = ROOT.Experimental.RNTupleWriteOptions


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

    def test_estimate_memory_usage(self):
        """Can estimate the memory usage of a model."""

        model = RNTupleModel.CreateBare()
        model.MakeField["int"]("i")
        model.MakeField["std::vector<std::vector<float>>"]("f")

        options = RNTupleWriteOptions()
        PageSize = 1234
        ClusterSize = 6789
        options.SetApproxUnzippedPageSize(PageSize)
        options.SetApproxZippedClusterSize(ClusterSize)

        Expected = 4 * 3 * PageSize + 3 * ClusterSize
        self.assertEqual(model.EstimateWriteMemoryUsage(options), Expected)


if __name__ == "__main__":
    unittest.main()
