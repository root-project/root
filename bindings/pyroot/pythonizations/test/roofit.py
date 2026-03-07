import unittest

import ROOT


class TestRooAbsCollection(unittest.TestCase):
    """
    Test for RooAbsCollection pythonizations.
    """

    num_elems = 3
    rooabsarg_list = [ROOT.RooStringVar(str(i), "", "") for i in range(num_elems)]
    tlist = ROOT.TList()

    # Setup
    @classmethod
    def setUpClass(cls):
        for elem in cls.rooabsarg_list:
            cls.tlist.Add(elem)

    @classmethod
    def tearDownClass(cls):
        # Clear TList before Python list deletes the objects
        cls.tlist.Clear()

    # Parametrized tests for RooAbsCollection child classes
    def _test_len(self, collection_class):
        c = collection_class(self.tlist)
        self.assertEqual(len(c), self.num_elems)
        self.assertEqual(len(c), c.getSize())

    def _test_addowned(self, collection_class):
        coll = collection_class()
        if True:
            x = ROOT.RooRealVar("x", "x", -10, 10)
            coll.addOwned(x)
        self.assertTrue("x" in coll)

    def _test_iterator(self, collection_class):
        a = ROOT.RooRealVar("a", "", 0)
        b = ROOT.RooRealVar("b", "", 0)

        l = collection_class(a, b)

        it = iter(l)

        self.assertEqual(next(it).GetName(), "a")
        self.assertEqual(next(it).GetName(), "b")

        with self.assertRaises(StopIteration):
            next(it)

    def _test_contains(self, collection_class):
        var0 = ROOT.RooRealVar("var0", "var0", 0)
        var1 = ROOT.RooRealVar("var1", "var1", 1)

        # The next variable has a duplicate name on purpose the check if it's
        # really the name that is used as the key in RooAbsCollections.
        var2 = ROOT.RooRealVar("var0", "var0", 0)

        coll = collection_class(var0)

        # expected behaviour
        self.assertTrue(var0 in coll)
        self.assertTrue("var0" in coll)

        self.assertTrue(not var1 in coll)
        self.assertTrue(not "var1" in coll)

        self.assertTrue(var2 in coll)
        self.assertTrue(not "var2" in coll)

        # ensure consistency with RooAbsCollection::find
        variables = [var0, var1, var2]

        for i, vptr in enumerate(variables):
            vname = "var" + str(i)
            self.assertEqual(coll.find(vptr) == vptr, vptr in coll)
            self.assertEqual(coll.find(vname) == vptr, vname in coll)

    def _test_getitem(self, collection_class):

        var0 = ROOT.RooRealVar("var0", "var0", 0)
        var1 = ROOT.RooRealVar("var1", "var1", 1)

        coll = collection_class(var0, var1)

        with self.assertRaises(TypeError):
            coll[1.5]

        with self.assertRaises(IndexError):
            coll[2]

        with self.assertRaises(IndexError):
            coll[-3]

        # check if negative indexing works
        self.assertEqual(coll[0], coll[-2])
        self.assertEqual(coll[1], coll[-1])

    # Tests
    def test_len_rooarglist(self):
        self._test_len(ROOT.RooArgList)

    def test_len_rooargset(self):
        self._test_len(ROOT.RooArgSet)

    def test_addowned_rooarglist(self):
        self._test_addowned(ROOT.RooArgList)

    def test_addowned_rooargset(self):
        self._test_addowned(ROOT.RooArgSet)

    def test_iterator_rooarglist(self):
        self._test_iterator(ROOT.RooArgList)

    def test_iterator_rooargset(self):
        self._test_iterator(ROOT.RooArgSet)

    def test_contains_rooarglist(self):
        self._test_contains(ROOT.RooArgList)

    def test_contains_rooargset(self):
        self._test_contains(ROOT.RooArgSet)

    def test_getitem_rooarglist(self):
        self._test_getitem(ROOT.RooArgList)
        # The RooArgList doesn't support string keys
        with self.assertRaises(TypeError):
            ROOT.RooArgList()["name"]

    def test_getitem_rooargset(self):
        self._test_getitem(ROOT.RooArgSet)
        # The RooArgSet supports string keys
        var0 = ROOT.RooRealVar("var0", "var0", 0)
        self.assertEqual(ROOT.RooArgSet(var0)["var0"], var0)

    def test_clone_collection(self):
        # Check explicitly that both overloads of addClone work.
        x = ROOT.RooRealVar("x", "x", 1.0)
        s1 = ROOT.RooArgSet(x)
        s2 = ROOT.RooArgSet()
        s3 = ROOT.RooArgSet()

        s2.addClone(s1)  # addClone(const RooAbsCollection& list)
        s3.addClone(x)  # addClone(const RooAbsArg& var)

    def test_rooargset_iter(self):
        """STL sequence iterator injected in RooAbsCollection, inherited by RooArgSet"""
        # ROOT-10606

        varA = ROOT.RooRealVar("a", "a", 0.0)
        varB = ROOT.RooRealVar("b", "b", 1.0)
        varSet = ROOT.RooArgSet(varA, varB)
        for var in varSet:
            var.getVal()


class RooAbsPdfFitTo(unittest.TestCase):
    """
    Test for the FitTo callable.
    """

    x = ROOT.RooRealVar("x", "x", -10, 10)
    mu = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
    sig = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gaussian", x, mu, sig)
    data = gauss.generate(ROOT.RooArgSet(x), 100)

    x.setRange("sideband", -10, 0)

    def _reset_initial_values(self):
        # after every fit, we have to reset the initial values because
        # otherwise the fit result is not considered the same
        self.mu.setVal(1.0)
        self.sig.setVal(1.0)
        self.mu.setError(0.0)
        self.sig.setError(0.0)

    def test_save(self):
        # test that kwargs can be passed
        # and lead to correct result
        gauss = self.gauss
        data = self.data
        self.assertEqual(
            gauss.fitTo(data, Save=False, PrintLevel=-1),
            gauss.fitTo(data, ROOT.RooFit.Save(ROOT.kFALSE), ROOT.RooFit.PrintLevel(-1)),
        )
        self.assertTrue(bool(gauss.fitTo(data, Save=True, PrintLevel=-1)))
        self._reset_initial_values()

    def test_wrong_kwargs(self):
        # test that AttributeError is raised
        # if keyword does not correspong to CmdArg
        gauss = self.gauss
        data = self.data
        self.assertRaises(AttributeError, gauss.fitTo, data, ThisIsNotACmgArg=True)
        self._reset_initial_values()

    def test_identical_result(self):
        # test that fitting with keyword arguments leads to the same result
        # as doing the same fit with passed ROOT objects
        gauss = self.gauss
        data = self.data
        res_1 = gauss.fitTo(data, Range="sideband", Save=True, PrintLevel=-1)
        self._reset_initial_values()
        res_2 = gauss.fitTo(data, ROOT.RooFit.Range("sideband"), ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
        self._reset_initial_values()
        self.assertTrue(res_1.isIdentical(res_2))

    def test_mixed_styles(self):
        # test that no error is causes if python style and cpp style
        # args are provided to fitto and that results are identical
        gauss = self.gauss
        data = self.data
        res_1 = gauss.fitTo(data, ROOT.RooFit.Range("sideband"), Save=True, PrintLevel=-1)
        self._reset_initial_values()
        res_2 = gauss.fitTo(data, ROOT.RooFit.Save(True), Range="sideband", PrintLevel=-1)
        self._reset_initial_values()
        self.assertTrue(res_1.isIdentical(res_2))


class RooAbsRealPlotOn(unittest.TestCase):
    """
    Test for the PlotOn callable.
    """

    x = ROOT.RooRealVar("x", "x", -10, 10)
    mu = ROOT.RooRealVar("mu", "mean", 1, -10, 10)
    sig = ROOT.RooRealVar("sig", "variance", 1, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gaussian", x, mu, sig)
    data = gauss.generate(ROOT.RooArgSet(x), 100)
    xframe = x.frame(ROOT.RooFit.Title("Gaussian pdf"))

    def test_frame(self):
        # test that kwargs can be passed
        # and lead to correct result
        r1 = self.gauss.plotOn(self.xframe, ROOT.RooFit.LineColor(ROOT.kRed))
        r2 = self.gauss.plotOn(self.xframe, LineColor=ROOT.kRed)

    def test_wrong_kwargs(self):
        # test that AttributeError is raised
        # if keyword does not correspong to CmdArg
        self.assertRaises(AttributeError, self.gauss.plotOn, self.xframe, ThisIsNotACmgArg=True)

    def test_binning(self):
        # test that fitting with keyword arguments leads to the same result
        # as doing the same plot with passed ROOT objects
        dtframe = self.x.frame(ROOT.RooFit.Range(-5, 5), ROOT.RooFit.Title("dt distribution with custom binning"))
        binning = ROOT.RooBinning(20, -5, 5)
        r1 = self.data.plotOn(dtframe, ROOT.RooFit.Binning(binning))
        r2 = self.data.plotOn(dtframe, Binning=binning)

    def test_data(self):
        # test that no error is causes if python style and cpp style
        # args are provided to plotOn and that results are identical
        frame = self.x.frame(ROOT.RooFit.Name("xframe"), ROOT.RooFit.Title("Red Curve"), ROOT.RooFit.Bins(20))
        res1_d1 = self.data.plotOn(frame, ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2))
        res2_d1 = self.data.plotOn(frame, DataError=ROOT.RooAbsData.SumW2)


class TestRooArgList(unittest.TestCase):
    """
    Test for RooArgList pythonizations.
    """

    def test_conversion_from_python_collection(self):

        # General check that the conversion from string or tuple works, using
        # constants to get compact test code.
        l1 = ROOT.RooArgList([1.0, 2.0, 3.0])
        self.assertEqual(len(l1), 3)

        l2 = ROOT.RooArgList((1.0, 2.0, 3.0))
        self.assertEqual(len(l2), 3)

        # Let's make sure that we can add two arguments with the same name to
        # the RooArgList. Here, we try to add the same RooConst two times. The
        # motivation for this test if the RooArgList is created via an
        # intermediate RooArgSet, which should not happen.
        l3 = ROOT.RooArgList([1.0, 2.0, 2.0])
        self.assertEqual(len(l3), 3)


# Necessary inside the "eval" call
RooArgSet = ROOT.RooArgSet
RooCmdArg = ROOT.RooCmdArg

x = ROOT.RooRealVar("x", "x", 1.0)
y = ROOT.RooRealVar("y", "y", 2.0)
z = ROOT.RooRealVar("z", "z", 3.0)


def args_equal(arg_1, arg_2):
    same = True

    same &= str(arg_1.GetName()) == str(arg_2.GetName())
    same &= str(arg_1.GetTitle()) == str(arg_2.GetTitle())

    for i in range(2):
        same &= arg_1.getInt(i) == arg_2.getInt(i)

    for i in range(2):
        same &= arg_1.getDouble(i) == arg_2.getDouble(i)

    for i in range(3):
        same &= str(arg_1.getString(i)) == str(arg_2.getString(i))

    same &= arg_1.procSubArgs() == arg_2.procSubArgs()
    same &= arg_1.prefixSubArgs() == arg_2.prefixSubArgs()

    for i in range(2):
        same &= arg_1.getObject(i) == arg_2.getObject(i)

    def set_equal(set_1, set_2):
        if set_1 == ROOT.nullptr and set_2 == ROOT.nullptr:
            return True
        if set_1 == ROOT.nullptr and set_2 != ROOT.nullptr:
            return False
        if set_1 != ROOT.nullptr and set_2 == ROOT.nullptr:
            return False

        if set_1.size() != set_2.size():
            return False

        return set_2.hasSameLayout(set_1)

    for i in range(2):
        same &= set_equal(arg_1.getSet(i), arg_2.getSet(i))

    return same


class TestRooArgList(unittest.TestCase):
    """
    Test for RooCmdArg pythonizations.
    """

    def test_constructor_eval(self):

        set_1 = ROOT.RooArgSet(x, y)
        set_2 = ROOT.RooArgSet(y, z)

        def do_test(*args):
            arg_1 = ROOT.RooCmdArg(*args)

            # The arg should be able to recreate itself by emitting the right
            # constructor code:
            arg_2 = eval(arg_1.constructorCode())

            self.assertTrue(args_equal(arg_1, arg_2))

        nullp = ROOT.nullptr

        # only fill the non-object fields:
        do_test("Test", -1, 3, 4.2, 4.7, "hello", "world", nullp, nullp, nullp, "s3", nullp, nullp)

        # RooArgSet tests:
        do_test("Test", -1, 3, 4.2, 4.7, "hello", "world", nullp, nullp, nullp, "s3", set_1, set_2)


class TestRooDataHistNumpy(unittest.TestCase):

    @staticmethod
    def _make_root_histo():
        # Create ROOT ROOT.TH1 filled with a Gaussian distribution
        hh = ROOT.TH1D("hh", "hh", 25, -10, 10)
        for i in range(100):
            hh.Fill(ROOT.gRandom.Gaus(0, 3), 0.5)
        return hh

    def test_to_numpy_and_from_numpy(self):
        """Test exporting to numpy and then importing back a RooDataHist."""

        hh = self._make_root_histo()

        # Declare observable x
        x = ROOT.RooRealVar("x", "x", -10, 10)

        datahist = ROOT.RooDataHist("data_hist", "data_hist", [x], Import=hh)

        hist, bin_edges = datahist.to_numpy()
        weights_squared_sum = datahist._weights_squared_sum()

        # We try both ways to pass bin edges: either via a full array, or with
        # the number of bins and the limits.
        datahist_1 = ROOT.RooDataHist.from_numpy(hist, [x], bins=bin_edges, weights_squared_sum=weights_squared_sum)
        datahist_2 = ROOT.RooDataHist.from_numpy(
            hist, [x], bins=[25], ranges=[(-10, 10)], weights_squared_sum=weights_squared_sum
        )

        def compare_to_ref(dh):
            import numpy as np

            hist_new, bin_edges_new = dh.to_numpy()
            np.testing.assert_allclose(hist_new, hist)
            np.testing.assert_allclose(bin_edges_new, bin_edges)
            np.testing.assert_allclose(dh._weights_squared_sum(), weights_squared_sum)

        compare_to_ref(datahist_1)
        compare_to_ref(datahist_2)


class RooDataHistPlotOn(unittest.TestCase):
    """
    Initially, this was a test for the pythonization that allowed
    RooDataHist to use the overloads of plotOn defined in RooAbsData.
    Currently, such functionality is automatically provided by Cppyy
    and ROOT meta: the overloads obtained with 'using' declarations
    are taken into account when calling a method.
    We keep this test to check that the aforementioned functionality
    works properly in a case that is important for RooFit.
    """

    # Helpers
    def create_hist_and_frame(self):
        # Inspired by the code of rf402_datahandling.py

        x = ROOT.RooRealVar("x", "x", -10, 10)
        y = ROOT.RooRealVar("y", "y", 0, 40)
        x.setBins(10)
        y.setBins(10)

        d = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x, y))
        for i in range(10):
            x.setVal(i / 2)
            y.setVal(i)
            d.add(ROOT.RooArgSet(x, y))

        dh = ROOT.RooDataHist("dh", "binned version of d", ROOT.RooArgSet(x, y), d)

        yframe = y.frame(Bins=10)

        return dh, yframe

    # Tests
    def test_overload1(self):
        dh, yframe = self.create_hist_and_frame()

        # Overload taken from RooAbsData
        # RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooCmdArg& arg1 = {},
        # const RooCmdArg& arg2 = {}, const RooCmdArg& arg3 = {},
        # const RooCmdArg& arg4 = {}, const RooCmdArg& arg5 = {},
        # const RooCmdArg& arg6 = {}, const RooCmdArg& arg7 = {},
        # const RooCmdArg& arg8 = {})
        res = dh.plotOn(yframe)
        self.assertEqual(type(res), ROOT.RooPlot)

    def test_overload2(self):
        dh, yframe = self.create_hist_and_frame()

        # Overload taken from RooAbsData
        # RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooLinkedList& cmdList)
        res = dh.plotOn(yframe, ROOT.RooLinkedList())
        self.assertEqual(type(res), ROOT.RooPlot)


class TestRooDataSet(unittest.TestCase):
    def test_createHistogram_decls(self):
        """RooDataSet::createHistogram overloads obtained with using decls."""

        import ROOT

        x = ROOT.RooRealVar("x", "x", -10, 10)
        mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
        sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
        gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

        data = gauss.generate(ROOT.RooArgSet(x), 10000)  # ROOT.RooDataSet
        h1d = data.createHistogram("myname", x)


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
        import numpy as np

        vars_in_data = data.get()
        x_in_data = vars_in_data["x"]
        cat_in_data = vars_in_data["cat"]

        for i in range(data.numEntries()):
            data.get(i)
            np.testing.assert_almost_equal(np_data["x"][i], x_in_data.getVal(), decimal=10)
            self.assertEqual(np_data["cat"][i], cat_in_data.getIndex())

    def test_to_numpy_and_from_numpy(self):
        """Test exporting to numpy and then importing back a non-weighted dataset."""
        import numpy as np

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
        import numpy as np

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
        import numpy as np

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

        import numpy as np

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


class TestRooGlobalFunc(unittest.TestCase):
    """
    Test for RooGlobalFunc pythonizations.
    """

    def test_color_codes(self):
        """Test that the color code pythonizations in the functions like
        RooFit.LineColor are working as they should.
        """

        def code(color):
            """Get the color code that will be obtained by a given argument
            passed to RooFit.LineColor.
            """
            return ROOT.RooFit.LineColor(color).getInt(0)

        # Check that general string to enum pythonization works
        self.assertEqual(code(ROOT.kRed), code("kRed"))

        # Check that matplotlib-style color strings work
        self.assertEqual(code(ROOT.kRed), code("r"))

        # Check that postfix operations applied to ROOT color codes work
        self.assertEqual(code(ROOT.kRed + 1), code("kRed+1"))

    def test_roodataset_link(self):
        """Test that the RooFit.Link() command argument works as expected in
        the RooDataSet constructor.
        Inspired by the reproducer code in GitHub issue #11469.
        """
        x = ROOT.RooRealVar("x", "", 0, 1)
        g = ROOT.RooGaussian("g", "", x, ROOT.RooFit.RooConst(0.5), ROOT.RooFit.RooConst(0.2))

        n_events = 1000

        data = g.generate({x}, NumEvents=n_events)

        sample = ROOT.RooCategory("cat", "cat")
        sample.defineType("cat_0")

        data_2 = ROOT.RooDataSet("data_2", "data_2", {x}, Index=sample, Link={"cat_0": data})

        self.assertEqual(data_2.numEntries(), n_events)

    def test_minimizer(self):
        """C++ object returned by RooFit::Minimizer should not be double deleted"""
        # ROOT-9516
        minimizer = ROOT.RooFit.Minimizer("Minuit2", "migrad")


class TestRooJSONFactoryWSTool(unittest.TestCase):
    """
    Test for RooJSONFactoryWSTool pythonizations.
    """

    def test_writedoc(self):

        ROOT.RooJSONFactoryWSTool.writedoc("roojsonfactorywstool_test_writedoc.tex")


class TestRooLinkedList(unittest.TestCase):
    """
    Tests for the RooLinkedList.
    """

    def test_roolinkedlist_iteration(self):
        # test if we can correctly iterate over a RooLinkedList, also in
        # reverse.

        roolist = ROOT.RooLinkedList()
        pylist = []

        n_elements = 3

        for i in range(n_elements):
            obj = ROOT.TNamed(str(i), str(i))
            ROOT.SetOwnership(obj, False)
            roolist.Add(obj)
            pylist.append(obj)

        self.assertEqual(len(roolist), n_elements)

        for i, obj in enumerate(roolist):
            self.assertEqual(str(i), obj.GetName())


class RooSimultaneous_test(unittest.TestCase):
    """
    Test for the pythonizations of RooSimultaneous.
    """

    # Tests
    def test_construction_from_dict(self):

        x = ROOT.RooRealVar("x", "x", -10, 10)

        # Define model for each sample
        model = ROOT.RooGaussian("model", "model", x, ROOT.RooFit.RooConst(1.0), ROOT.RooFit.RooConst(1.0))
        model_ctl = ROOT.RooGaussian("model_ctl", "model_ctl", x, ROOT.RooFit.RooConst(-1.0), ROOT.RooFit.RooConst(1.0))

        # Define category to distinguish physics and control samples events
        sample = ROOT.RooCategory("sample", "sample")
        sample.defineType("physics")
        sample.defineType("control")

        # Construct the RooSimultaneous
        sim_pdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", {"physics": model, "control": model_ctl}, sample)

        # Verify that the PDF ends up with the right PDFs in the right categories
        self.assertEqual(sim_pdf.getPdf("physics").GetName(), "model")
        self.assertEqual(sim_pdf.getPdf("control").GetName(), "model_ctl")


class RooWorkspace_test(unittest.TestCase):
    """
    Test for the pythonizations of RooWorkspace.
    """

    # Setup
    def setUp(self):
        self.x = ROOT.RooRealVar("x", "x", 1.337, 0, 10)
        self.ws = ROOT.RooWorkspace("ws", "A workspace")

    # Tests
    def test_import(self):
        self.ws.Import(self.x)
        x = self.ws.var("x")
        self.assertEqual(x.GetName(), "x")
        self.assertEqual(x.getVal(), self.x.getVal())

    def test_import_with_arg(self):
        # Prepare workspace with variables and a PDF
        self.ws.Import(self.x)
        rename = ROOT.RooFit.RenameAllVariables("exp")
        exp = ROOT.RooExponential("exp", "exp", self.x, self.x)
        self.ws.Import(exp, rename)

        # Test that rename argument has worked
        x = self.ws.var("x_exp")
        self.assertEqual(x.getVal(), self.x.getVal())
        pdf = self.ws.pdf("exp")
        self.assertGreater(pdf.getVal(), 0)

    def test_import_argset(self):
        argSet = ROOT.RooArgSet(self.x)
        self.ws.Import(argSet)
        x = self.ws.arg("x")
        self.assertEqual(x.GetName(), "x")
        self.assertEqual(x.getVal(), self.x.getVal())

    def test_set_item_using_string(self):
        # Test to check if new variables are created
        self.ws["z"] = "[3]"
        self.assertEqual(self.ws["z"].GetName(), "z")
        self.assertEqual(self.ws["z"].getVal(), 3.0)

        # Test to check if new p.d.f.s are created
        self.ws["gauss"] = "Gaussian(x[0.0, 10.0], mu[5.0], sigma[2.0, 0.01, 10.0])"
        self.assertEqual(self.ws["gauss"].getX(), self.ws["x"])
        self.assertEqual(self.ws["gauss"].getMean(), self.ws["mu"])
        self.assertEqual(self.ws["gauss"].getSigma(), self.ws["sigma"])

    def test_set_item_using_dictionary(self):
        ws = ROOT.RooWorkspace()

        # Test to check if new variables are created
        ws["x"] = dict({"min": 0.0, "max": 10.0})
        self.assertEqual(ws["x"].getMax(), 10.0)
        self.assertEqual(ws["x"].getMin(), 0.0)

        # Test to check if new functions are created
        ws["m1"] = dict({"max": 5, "min": -5, "value": 0})
        ws["m2"] = dict({"max": 5, "min": -5, "value": 1})
        ws["mean"] = dict({"type": "sum", "summands": ["m1", "m2"]})
        self.assertEqual(ws["mean"].GetName(), "mean")
        self.assertEqual(ws["mean"].getVal(), 1.0)

        # Test to check if new p.d.f.s are created
        ws["sigma"] = dict({"value": 2, "min": 0.1, "max": 10.0})
        ws["gauss"] = dict({"mean": "mean", "sigma": "sigma", "type": "gaussian_dist", "x": "x"})
        self.assertEqual(ws["gauss"].getX(), ws["x"])
        self.assertEqual(ws["gauss"].getMean(), ws["mean"])
        self.assertEqual(ws["gauss"].getSigma(), ws["sigma"])


if __name__ == "__main__":
    unittest.main()
