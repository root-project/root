import numpy
import pytest
import ROOT


class TestAsNumpy:
    """
    Tests of proper functionality of the AsNumpy action across different
    distributed configurations.
    """

    @pytest.mark.parametrize("nfiles", [1, 3, 7])
    @pytest.mark.parametrize("nparts", [1, 2, 3, 7, 8, 15, 16, 21])
    @pytest.mark.parametrize("datasource", ["ttree", "rntuple"])
    def test_clone_asnumpyresult(self, payload, nfiles, nparts, datasource):
        """
        Test that the correct values of the numpy array are retrieved from
        distributed execution irrespective of the number of partitions.
        """

        datasetname = "Events"
        filename = f"../data/{datasource}/distrdf_roottest_check_cloned_actions_asnumpy.root"
        inputfiles = [filename] * nfiles
        connection, _ = payload
        distrdf = ROOT.RDataFrame(datasetname, inputfiles, executor=connection, npartitions=nparts)

        localdf = ROOT.RDataFrame("Events", inputfiles)

        vals_distrdf = distrdf.AsNumpy(["event"])
        vals_localdf = localdf.AsNumpy(["event"])

        # Distributed mode does not guarantee the order of execution of the tasks
        # thus the output numpy array is unsorted. We also sort the output array
        # of the local execution so that in case we test with multiple files
        # the values of the arrays can be properly aligned (otherwise it would
        # always fail).
        assert all(numpy.sort(vals_localdf["event"]) == numpy.sort(vals_distrdf["event"]))


if __name__ == "__main__":
    pytest.main(args=[__file__])
