import unittest

import numpy as np
import pandas
import ROOT
import xgboost

np.random.seed(1234)


def create_dataset(num_events, num_features, num_outputs, dtype=np.float32):
    x = np.random.normal(0.0, 1.0, (num_events, num_features)).astype(dtype=dtype)
    if num_outputs == 1:
        y = np.random.normal(0.0, 1.0, (num_events)).astype(dtype=dtype)
    else:
        y = np.random.choice(
            a=range(num_outputs), size=(num_events), p=[1.0 / float(num_outputs)] * num_outputs
        ).astype(dtype=dtype)
    return x, y


def _test_XGBBinary(label):
    """
    Compare response of XGB classifier and TMVA tree inference system.
    """
    x, y = create_dataset(1000, 10, 2)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGBBinary{}.root".format(label), num_inputs=10)
    bdt = ROOT.TMVA.Experimental.RBDT("myModel", "testXGBBinary{}.root".format(label))

    y_xgb = xgb.predict_proba(x)[:, 1].squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


def _test_XGBRegression(label):
    """
    Compare response of XGB regressor and TMVA tree inference system.
    """
    n_samples = 1000
    n_features = 10
    x, y = create_dataset(n_samples, n_features, 1)
    # Other than in the XGBBinary test, we're passing the training features via
    # a pandas DataFrame this time. In that case, XGBoost will define custom
    # feature names according to the column names in the dataframe, and we can
    # test the case where the feature names in the .txt dump are not the
    # default "f0", "f1", "f2", etc.
    df_x = pandas.DataFrame({f"myfeature_{i}": x[:, i] for i in range(n_features)})
    assert len(x) == len(df_x)
    xgb = xgboost.XGBRegressor(n_estimators=1, max_depth=3)
    xgb.fit(df_x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGBRegression{}.root".format(label), num_inputs=10)
    bdt = ROOT.TMVA.Experimental.RBDT("myModel", "testXGBRegression{}.root".format(label))

    y_xgb = xgb.predict(x).squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


def _test_XGBMulticlass(label):
    """
    Compare response of XGB multiclass and TMVA tree inference system.
    """
    x, y = create_dataset(1000, 10, 3)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    ROOT.TMVA.Experimental.SaveXGBoost(xgb, "myModel", "testXGBMulticlass{}.root".format(label), num_inputs=10)
    bdt = ROOT.TMVA.Experimental.RBDT("myModel", "testXGBMulticlass{}.root".format(label))

    y_xgb = xgb.predict_proba(x)
    y_bdt = bdt.Compute(x)
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


class RBDT(unittest.TestCase):
    """
    Test RBDT interface
    """

    def test_XGBBinary_default(self):
        """
        Test model trained with binary XGBClassifier.
        """
        _test_XGBBinary("default")

    def test_XGBMulticlass_default(self):
        """
        Test model trained with multiclass XGBClassifier.
        """
        if xgboost.__version__ >= "3.1.0":
            self.skipTest("We don't support multiclassification with xgboost>=3.1.0 yet")
        _test_XGBMulticlass("default")

    def test_XGBRegression_default(self):
        """
        Test model trained with XGBRegressor.
        """
        _test_XGBRegression("default")


if __name__ == "__main__":
    unittest.main()
