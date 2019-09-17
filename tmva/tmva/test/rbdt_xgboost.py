import unittest
import ROOT
import numpy as np
np.random.seed(1234)
import xgboost


def create_dataset(num_events, num_features, num_outputs, dtype=np.float32):
    x = np.random.normal(0.0, 1.0, (num_events, num_features)).astype(dtype=dtype)
    if num_outputs == 1:
        y = np.random.normal(0.0, 1.0, (num_events)).astype(dtype=dtype)
    else:
        y = np.random.choice(a=range(num_outputs), size=(num_events), p=[1.0 / float(num_outputs)] * num_outputs).astype(dtype=dtype)
    return x, y


def _test_XGBBinary(backend, label):
    """
    Compare response of XGB classifier and TMVA tree inference system for a given backend
    """
    x, y = create_dataset(1000, 10, 2)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGBBinary{}.root".format(label))
    bdt = ROOT.TMVA.Experimental.RBDT[backend]("myModel", "testXGBBinary{}.root".format(label))

    y_xgb = xgb.predict_proba(x)[:, 1].squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


def _test_XGBRegression(backend, label):
    """
    Compare response of XGB regressor and TMVA tree inference system for a given backend
    """
    x, y = create_dataset(1000, 10, 1)
    xgb = xgboost.XGBRegressor(n_estimators=1, max_depth=3)
    xgb.fit(x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGBRegression{}.root".format(label), num_inputs=10)
    bdt = ROOT.TMVA.Experimental.RBDT[backend]("myModel", "testXGBRegression{}.root".format(label))

    y_xgb = xgb.predict(x).squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


def _test_XGBMulticlass(backend, label):
    """
    Compare response of XGB multiclass and TMVA tree inference system for a given backend
    """
    x, y = create_dataset(1000, 10, 3)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGBMulticlass{}.root".format(label))
    bdt = ROOT.TMVA.Experimental.RBDT[backend]("myModel", "testXGBMulticlass{}.root".format(label))

    y_xgb = xgb.predict_proba(x)
    y_bdt = bdt.Compute(x)
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


class RBDT(unittest.TestCase):
    """
    Test RBDT interface
    """

    def test_XGBBinary_default(self):
        """
        Test default backend for model trained with binary XGBClassifier
        """
        _test_XGBBinary("", "default")

    def test_XGBBinary_branchless(self):
        """
        Test BranchlessForest backend for model trained with binary XGBClassifier
        """
        _test_XGBBinary("TMVA::Experimental::BranchlessForest<float>", "branchlessForest")

    def test_XGBMulticlass_default(self):
        """
        Test default backend for model trained with multiclass XGBClassifier
        """
        _test_XGBMulticlass("", "default")

    def test_XGBMulticlass_branchless(self):
        """
        Test BranchlessForest backend for model trained with multiclass XGBClassifier
        """
        _test_XGBMulticlass("TMVA::Experimental::BranchlessForest<float>", "branchlessForest")

    def test_XGBRegression_default(self):
        """
        Test default backend for model trained with XGBRegressor
        """
        _test_XGBRegression("", "default")

    def test_XGBRegression_branchless(self):
        """
        Test BranchlessForest backend for model trained with XGBRegressor
        """
        _test_XGBRegression("TMVA::Experimental::BranchlessForest<float>", "branchlessForest")


if __name__ == '__main__':
    unittest.main()
