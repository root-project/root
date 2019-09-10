import unittest
import ROOT
import numpy as np
np.random.seed(1234)
import xgboost


def create_dataset(num_events, num_features, dtype=np.float32):
    x = np.random.normal(0.0, 1.0, (num_events, num_features)).astype(dtype=dtype)
    y = np.random.choice(a=[0, 1], size=(num_events), p=[0.5, 0.5]).astype(dtype=dtype)
    return x, y


def _test_XGBClassifier(backend, label):
    """
    Compare response of XGB and TMVA tree inference system for a given backend
    """
    x, y = create_dataset(1000, 10)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGB{}.root".format(label))
    bdt = ROOT.TMVA.Experimental.RBDT[backend]("myModel", "testXGB{}.root".format(label))

    y_xgb = xgb.predict_proba(x)[:, 1].squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


class RBDT(unittest.TestCase):
    """
    Test RBDT interface
    """

    def test_XGBClassifier_default(self):
        """
        Test default backend for model trained with XGBClassifier
        """
        _test_XGBClassifier("", "default")

    def test_XGBClassifier_default(self):
        """
        Test BranchlessForest backend for model trained with XGBClassifier
        """
        _test_XGBClassifier("TMVA::Experimental::BranchlessForest<float>", "branchlessForest")


if __name__ == '__main__':
    unittest.main()
