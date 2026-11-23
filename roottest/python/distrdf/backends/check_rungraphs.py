import pytest
import ROOT
import ROOT._distrdf


class TestRunGraphs:
    """Tests usage of RunGraphs function with Dask backend"""

    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_rungraphs_dask_3histos(self, payload, datasource):
        """
        Submit three different Dask RDF graphs concurrently
        """
        # Create a test file for processing
        treename = "tree"
        filename = f"../data/{datasource}/distrdf_roottest_check_rungraphs.root"
        nentries = 10000
        connection, _ = payload
        df = ROOT.RDataFrame(treename, filename, executor=connection, npartitions=2)
        histoproxies = [
            df.Histo1D((col, col, 1, 40, 45), col)
            for col in ["b1", "b2", "b3"]
        ]

        # Before triggering the computation graphs values are None
        for proxy in histoproxies:
            assert proxy.proxied_node.value is None

        ROOT.RDF.RunGraphs(histoproxies)

        # After RunGraphs all histograms are correctly assigned to the
        # node objects
        for proxy in histoproxies:
            histo = proxy.proxied_node.value
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == nentries
            assert histo.GetMean() == 42

    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_rungraphs_dask_no_premature_return(self, payload, datasource):
        """
        Submit three independent Dask RDF graphs concurrently and check that
        RunGraphs does not return until all of them have been fully processed.
        """
        # Create a test file for processing
        treename = "tree"
        filename = f"../data/{datasource}/distrdf_roottest_check_rungraphs.root"
        nentries = 10000
        connection, _ = payload
        # Three independent dataframes, i.e. three independent computation graphs
        histoproxies = [
            ROOT.RDataFrame(treename, filename, executor=connection, npartitions=2).Histo1D((col, col, 1, 40, 45), col)
            for col in ["b1", "b2", "b3"]
        ]

        # Before triggering the computation graphs values are None
        for proxy in histoproxies:
            assert proxy.proxied_node.value is None

        ROOT.RDF.RunGraphs(histoproxies)

        # RunGraphs must not return before all computation graphs are done:
        # every value must already be in place at this point, without the need
        # to trigger any further event loop with GetValue()
        for proxy in histoproxies:
            assert proxy.proxied_node.value is not None, (
                "RunGraphs returned before all computation graphs were fully processed"
            )

        # After RunGraphs all histograms are correctly assigned to the
        # node objects
        for proxy in histoproxies:
            histo = proxy.GetValue()
            assert isinstance(histo, ROOT.TH1D)
            assert histo.GetEntries() == nentries
            assert histo.GetMean() == 42


if __name__ == "__main__":
    pytest.main(args=[__file__])
