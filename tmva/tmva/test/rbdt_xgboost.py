import os
import tempfile
import unittest

import numpy as np
import pandas
import ROOT
import xgboost

np.random.seed(1234)


def save_xgboost(xgb, key_name, output_path):
    """Serialize the model to XGBoost's native JSON format and convert it to an
    RBDT in a ROOT file via the C++ TMVA::Experimental::SaveXGBoost."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_json:
        json_path = tmp_json.name
    try:
        xgb.get_booster().save_model(json_path)
        ROOT.TMVA.Experimental.SaveXGBoost(json_path, key_name, output_path)
    finally:
        os.remove(json_path)


def create_dataset(num_events, num_features, num_outputs, dtype=np.float32):
    x = np.random.normal(0.0, 1.0, (num_events, num_features)).astype(dtype=dtype)
    if num_outputs == 1:
        y = np.random.normal(0.0, 1.0, (num_events)).astype(dtype=dtype)
    else:
        y = np.random.choice(
            a=range(num_outputs), size=(num_events), p=[1.0 / float(num_outputs)] * num_outputs
        ).astype(dtype=dtype)
    return x, y


def _test_XGBBinary(output_path):
    """
    Compare response of XGB classifier and TMVA tree inference system.
    """
    x, y = create_dataset(1000, 10, 2)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    save_xgboost(xgb, "myModel", output_path)
    bdt = ROOT.TMVA.Experimental.RBDT("myModel", output_path)

    y_xgb = xgb.predict_proba(x)[:, 1].squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


def _test_XGBRegression(output_path):
    """
    Compare response of XGB regressor and TMVA tree inference system.
    """
    n_samples = 1000
    n_features = 10
    x, y = create_dataset(n_samples, n_features, 1)
    # Other than in the XGBBinary test, we're passing the training features via
    # a pandas DataFrame this time. In that case, XGBoost will define custom
    # feature names according to the column names in the dataframe, and we can
    # test the case where the feature names are not the default "f0", "f1",
    # "f2", etc.
    df_x = pandas.DataFrame({f"myfeature_{i}": x[:, i] for i in range(n_features)})
    assert len(x) == len(df_x)
    xgb = xgboost.XGBRegressor(n_estimators=1, max_depth=3)
    xgb.fit(df_x, y)
    save_xgboost(xgb, "myModel", output_path)
    bdt = ROOT.TMVA.Experimental.RBDT("myModel", output_path)

    y_xgb = xgb.predict(x).squeeze()
    y_bdt = bdt.Compute(x).squeeze()
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


def _test_XGBMulticlass(output_path):
    """
    Compare response of XGB multiclass and TMVA tree inference system.
    """
    x, y = create_dataset(1000, 10, 3)
    xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
    xgb.fit(x, y)
    save_xgboost(xgb, "myModel", output_path)
    bdt = ROOT.TMVA.Experimental.RBDT("myModel", output_path)

    y_xgb = xgb.predict_proba(x)
    y_bdt = bdt.Compute(x)
    np.testing.assert_array_almost_equal(y_xgb, y_bdt)


class RBDT(unittest.TestCase):
    """
    Test RBDT interface
    """

    def setUp(self):
        # Keep all model files in a temporary directory so the test leaves no
        # spurious artifacts behind, regardless of the working directory.
        self._tmpdir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self._tmpdir.name, "model.root")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_XGBBinary_default(self):
        """
        Test model trained with binary XGBClassifier.
        """
        _test_XGBBinary(self.output_path)

    def test_XGBMulticlass_default(self):
        """
        Test model trained with multiclass XGBClassifier.
        """
        if xgboost.__version__ >= "3.1.0":
            self.skipTest("We don't support multiclassification with xgboost>=3.1.0 yet")
        _test_XGBMulticlass(self.output_path)

    def test_XGBRegression_default(self):
        """
        Test model trained with XGBRegressor.
        """
        _test_XGBRegression(self.output_path)


if __name__ == "__main__":
    unittest.main()
