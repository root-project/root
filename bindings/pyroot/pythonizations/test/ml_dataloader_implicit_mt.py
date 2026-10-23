import pytest
import ROOT


class TestDataLoaderImplicitMT:
    file_name = "dataloader_imt.root"
    tree_name = "mytree"

    def teardown_method(self):
        ROOT.DisableImplicitMT()

    @pytest.mark.skipif(
        "imt" not in ROOT.gROOT.GetConfigFeatures(), reason="ROOT was built without implicit multi-threading"
    )
    def test01_implicit_mt_not_supported(self):
        ROOT.RDataFrame(10).Define("b1", "(int) rdfentry_").Snapshot(self.tree_name, self.file_name)
        ROOT.EnableImplicitMT(2)
        df = ROOT.RDataFrame(self.tree_name, self.file_name)

        with pytest.raises(RuntimeError, match="Cannot use RDataLoader with implicit multi-threading enabled"):
            ROOT.Experimental.ML.RDataLoader(df, batch_size=2).as_numpy()


if __name__ == "__main__":
    raise SystemExit(pytest.main(args=[__file__]))
