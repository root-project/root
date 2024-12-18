import unittest

import ROOT

import numpy as np


class TestRooDataSetNumpy(unittest.TestCase):
    def _create_dataset(self):
        x = ROOT.RooRealVar("x", "x", 0, 10)
        cat = ROOT.RooCategory("cat", "cat")
        cat.defineType("minus", -1)
        cat.defineType("plus", +1)
        mean = ROOT.RooRealVar("mean", "mean of gaussian", 5, 0, 10)
        sigma = ROOT.RooRealVar("sigma", "width of gaussian", 2, 0.1, 10)
        gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)
        data = gauss.generate((x, cat), 100)
        return data, x, cat

    def test_to_numpy_basic(self):
        """Basic test with a real value and a category."""
        data, _, _ = self._create_dataset()
        self.assertEqual(set(data.to_numpy().keys()), {"x", "cat"})

    def test_to_numpy_weighted(self):
        """Test with a weighted dataset."""
        import numpy as np

        x = ROOT.RooRealVar("x", "x", -10, 10)
        myweight = ROOT.RooRealVar("myweight", "myweight", 0, 400)

        columns = (x, myweight)

        data = ROOT.RooDataSet("data", "data", columns, WeightVar=myweight)
        for j in range(100):
            x.setVal(np.random.normal())
            myweight.setVal(10 + np.random.normal())
            data.add(columns, myweight.getVal())

        self.assertEqual(set(data.to_numpy().keys()), {"x", "myweight"})

    def test_to_numpy_derived_weight(self):
        """Test if the optional computation of derived weights works."""

        data, x, _ = self._create_dataset()

        # Create a data set with a derived weight
        wFunc = ROOT.RooFormulaVar("w", "event weight", "(x*x+10)", [x])
        w = data.addColumn(wFunc)
        wdata = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data.get(), Import=data, WeightVar=w.GetName())

        self.assertEqual(set(wdata.to_numpy().keys()), {"x", "cat", "w"})

    def _check_value_equality(self, data, np_data):
        vars_in_data = data.get()
        x_in_data = vars_in_data["x"]
        cat_in_data = vars_in_data["cat"]

        for i in range(data.numEntries()):
            data.get(i)
            np.testing.assert_almost_equal(np_data["x"][i], x_in_data.getVal(), decimal=10)
            self.assertEqual(np_data["cat"][i], cat_in_data.getIndex())

    def test_to_numpy_and_from_numpy(self):
        """Test exporting to numpy and then importing back a non-weighted dataset."""

        data, x, cat = self._create_dataset()

        np_data = data.to_numpy()

        data_2 = ROOT.RooDataSet.from_numpy(np_data, (x, cat), name="data_2", title="data_2")

        np_data_2 = data_2.to_numpy()

        np.testing.assert_almost_equal(data_2.sumEntries(), data.sumEntries(), decimal=10)
        self.assertEqual(set(np_data.keys()), set(np_data_2.keys()))

        for key in np_data.keys():
            self.assertEqual(len(np_data[key]), len(np_data_2[key]))
            self.assertEqual(np_data[key][0], np_data_2[key][0])
            self.assertEqual(np_data[key][-1], np_data_2[key][-1])

        self._check_value_equality(data, np_data)
        self._check_value_equality(data, np_data_2)

    def test_to_numpy_and_from_numpy_weighted(self):
        """Test exporting to numpy and then importing back a weighted dataset."""

        data, x, cat = self._create_dataset()
        wvar = ROOT.RooRealVar("w", "w", 0, 110)

        # Construct formula to calculate (fake) weight for events
        wFunc = ROOT.RooFormulaVar("w", "event weight", "(x*x+10)", [x])

        # Add column with variable w to previously generated dataset
        w = data.addColumn(wFunc)

        # Instruct dataset wdata to use w as event weight and not observable
        wdata = ROOT.RooDataSet(data.GetName(), data.GetTitle(), data.get(), Import=data, WeightVar=w.GetName())

        np_data = wdata.to_numpy()

        wdata_2 = ROOT.RooDataSet.from_numpy(np_data, (x, cat, wvar), name="wdata_2", title="wdata_2", weight_name="w")

        np_data_2 = wdata_2.to_numpy()

        np.testing.assert_almost_equal(wdata_2.sumEntries(), wdata.sumEntries(), decimal=10)
        self.assertEqual(set(np_data.keys()), set(np_data_2.keys()))

        for key in np_data.keys():
            self.assertEqual(len(np_data[key]), len(np_data_2[key]))
            self.assertEqual(np_data[key][0], np_data_2[key][0])
            self.assertEqual(np_data[key][-1], np_data_2[key][-1])

        self._check_value_equality(data, np_data)
        self._check_value_equality(data, np_data_2)

    def test_ignoring_out_of_range(self):
        """Test that rows with out-of-range values are skipped, both for
        real-valued columns and categories.
        """

        n_events = 100
        # Dataset with "x" randomly distributed between -3 and 3, and "cat"
        # being either -1, 0, or +1.
        data = {
            "x": np.random.rand(n_events) * 6.0 - 3.0,
            "cat": np.random.randint(3, size=n_events) - 1,
        }

        # The RooFit variable "x" is only defined from -1 to 2, and the
        # category doesn't have the 0-state.
        x = ROOT.RooRealVar("x", "x", 0.0, -2.0, 2.0)
        cat = ROOT.RooCategory("cat", "cat")
        cat.defineType("minus", -1)
        cat.defineType("plus", +1)

        # Use manual loop because we had some problems with numpys boolean
        # comparisons in the past (see GitHub issue #12162).
        n_in_range = 0
        for i in range(n_events):
            in_x_range = data["x"][i] <= x.getMax() and data["x"][i] >= x.getMin()
            in_cat_range = (data["cat"][i] == -1) or (data["cat"][i] == +1)
            is_in_range = in_x_range and in_cat_range
            if is_in_range:
                n_in_range = n_in_range + 1

        dataset_numpy = ROOT.RooDataSet.from_numpy(data, {x, cat}, name="dataSetNumpy")

        self.assertEqual(dataset_numpy.numEntries(), n_in_range)

    def test_non_contiguous_arrays(self):
        """Test whether the import also works with non-contiguous arrays.
        Covers GitHub issue #13605.
        """

        import itertools

        obs_1 = ROOT.RooRealVar("obs_1", "obs_1", 70, 70, 190)
        obs_1.setBins(30)
        obs_2 = ROOT.RooRealVar("obs_2", "obs_2", 100, 100, 180)
        obs_2.setBins(80)

        val_obs_1 = []
        val_obs_2 = []
        for i in range(obs_1.numBins()):
            obs_1.setBin(i)
            val_obs_1.append(obs_1.getVal())
        for i in range(obs_2.numBins()):
            obs_2.setBin(i)
            val_obs_2.append(obs_2.getVal())

        # so that all combination of values are in the dataset
        val_cart_product = np.array(list(itertools.product(val_obs_1, val_obs_2)))
        data = {"obs_1": val_cart_product[:, 0], "obs_2": val_cart_product[:, 1]}

        # To make sure the array is really not C-contiguous
        assert data["obs_1"].flags["C_CONTIGUOUS"] == False

        dataset = ROOT.RooDataSet.from_numpy(data, ROOT.RooArgSet(obs_1, obs_2))

        data_roundtripped = dataset.to_numpy()

        np.testing.assert_equal(data["obs_1"], data_roundtripped["obs_1"])
        np.testing.assert_equal(data["obs_2"], data_roundtripped["obs_2"])


if __name__ == "__main__":
    unittest.main()
