import shlex
import sys

import pytest
import ROOT


class TestFromSpec:
    # RDataFrame is reading files containing CMSSW classes in this test, which will trigger a warning from TClass about
    # missing class dictionaries. The warning is only triggered once for the entire duration of the process. Within the
    # same process, we may be running the same test multiple times (once per parametrized backend in use). So we can't
    # use programmatic warning catching e.g. with pytest.warns because it would always fail on consecutive runs of this
    # test. We filter out that warning specifically instead
    @pytest.mark.filterwarnings("ignore:no dictionary for class")
    def test_fromspec_different_trees(self, payload):
        """
        Test usage of FromSpec function when each sample has different trees
        """

        connection, _ = payload

        jsonfile = "../data/ttree/spec_differenttrees.json"

        df = ROOT.RDF.Experimental.FromSpec(jsonfile, executor=connection)
        df_checkfilt = df.FilterAvailable("nElectron").Filter("nElectron > 2")
        df_new = df.DefinePerSample("lum", 'rdfsampleinfo_.GetD("lum")')
        df_filtered = df_new.Filter("lum == 100.")
        df_filtered_two = df_new.Filter("lum == 200.")

        df_local = ROOT.RDF.Experimental.FromSpec(jsonfile)
        df_new_local = df_local.DefinePerSample("lum", 'rdfsampleinfo_.GetD("lum")')
        df_filtered_local = df_new_local.Filter("lum == 100.")
        df_filtered_two_local = df_new_local.Filter("lum == 200.")

        df_checkfilt_count = df_checkfilt.Count()
        df_filtered_count = df_filtered.Count()
        df_filtered_local_count = df_filtered_local.Count()
        df_filtered_two_count = df_filtered_two.Count()
        df_filtered_two_local_count = df_filtered_two_local.Count()

        assert df_checkfilt_count.GetValue() == 1683
        assert df_filtered_count.GetValue() == 11000
        assert df_filtered_count.GetValue() == df_filtered_local_count.GetValue()

        assert df_filtered_two_count.GetValue() == 13020
        assert df_filtered_two_local_count.GetValue() == df_filtered_two_local_count.GetValue()


if __name__ == "__main__":
    # The call to sys.exit is needed otherwise CTest would just ignore the
    # results returned by pytest, even in case of errors.
    # We ignore ResourceWarning about unclosed socket because of https://issues.apache.org/jira/browse/SPARK-38659 which
    # has been fixed by https://github.com/apache/spark/pull/53200 and https://github.com/apache/spark/pull/53203 which
    # may not be available in all test runner configurations
    sys.exit(
        pytest.main(args=shlex.split(f'{__file__} -x -vvv -Werror -Wignore:"unclosed <socket.socket":ResourceWarning'))
    )
