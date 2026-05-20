import pytest

import ROOT


class TestMissingValues:
    """Tests of dealing with missing values in the input dataset."""

    def test_defaults_and_missing(self, payload):
        """
        Test DefaultValueFor, FilterAvailable, FilterMissing operations
        string of operations.
        """
        filenames = [
            # 10k entries, defining b1, b2, b3 (Int_t), all always equal to 42
            f"../data/ttree/distrdf_roottest_check_rungraphs.root",
            # 100 entries defining 'v' (Double_t)
            f"../data/ttree/distrdf_roottest_check_reducer_merge_1.root",
        ]
        connection, _ = payload
        df = ROOT.RDataFrame("tree", filenames, executor=connection)
        c10k = df.FilterAvailable("b1").Count()
        c100 = df.FilterAvailable("v").Count()
        c100b = df.FilterMissing("b1").Count()
        cD10k = df.DefaultValueFor("b1", 40).Filter("b1 == 42").Count()
        cD10100 = df.DefaultValueFor("b1", 42).Filter("b1 == 42").Count()
        sV = df.DefaultValueFor("v", 0.1).Sum("v")
        assert c10k.GetValue() == 10000, f"{c10k.GetValue()=}"
        assert c100.GetValue() == 100, f"{c100.GetValue()=}"
        assert c100b.GetValue() == 100, f"{c100b.GetValue()=}"
        assert cD10k.GetValue() == 10000, f"{cD10k.GetValue()=}"
        assert cD10100.GetValue() == 10100, f"{cD10100.GetValue()=}"
        assert sV.GetValue() == 5950.0, f"{sV.GetValue()=}"


if __name__ == "__main__":
    pytest.main(args=[__file__])
