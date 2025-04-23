import pytest

import ROOT

import DistRDF

class TestFromSpec:

    def test_fromspec(self, payload):
        """
        Test usage of FromSpec function
        """
        
        jsonfile = "../data/ttree/spec.json"
        connection, _ = payload
        df = ROOT.RDF.Experimental.FromSpec(jsonfile, executor=connection, npartitions=2)

        df_filt = df.Filter("b1 > 42")
        
        nentries = 1000

        assert df_filt.Count().GetValue() == nentries
        assert df_filt.Mean("b1").GetValue() == 50


if __name__ == "__main__":
    pytest.main(args=[__file__])