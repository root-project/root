import math
import pytest
import ROOT


class TestDeclare:
    """
    Check if the distributed c++ code declaration works as expected.
    """

    def _check_rdf_histos_1(self, rdf):
        # This filters out all numbers less than 5
        rdf_filtered = rdf.Filter("check_number_less_than_five(rdfentry_)")
        histo = rdf_filtered.Histo1D(("name", "title", 10, 0, 10), "rdfentry_")

        # The expected results after filtering
        # The actual set of numbers required after filtering
        required_numbers = range(5)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(
            sum((x - required_mean) ** 2 for x in required_numbers) / required_size
        )

        # Compare the sizes of equivalent set of numbers
        assert histo.GetEntries() == required_size
        # Compare the means of equivalent set of numbers
        assert histo.GetMean() == required_mean
        # Compare the standard deviations of equivalent set of numbers
        assert histo.GetStdDev() == required_stdDev

    def _check_rdf_histos_2(self, rdf):
        rdf_filtered = rdf.Filter("check_number_less_than_four(rdfentry_)")
        histo = rdf_filtered.Histo1D(("name", "title", 10, 0, 10), "rdfentry_")

        required_numbers = range(4)
        required_size = len(required_numbers)
        required_mean = sum(required_numbers) / float(required_size)
        required_stdDev = math.sqrt(
            sum((x - required_mean) ** 2 for x in required_numbers) / required_size
        )

        assert histo.GetEntries() == required_size
        assert histo.GetMean() == required_mean
        assert histo.GetStdDev() == required_stdDev

    def _mydeclare_1(self, rdf):
        ROOT.RDF.Distributed.DistributeCppCode(
            """                                                                                      
            bool check_number_less_than_five(int num){
                return num < 5;
            }  
            """
        )

    def _mydeclare_2(self, rdf):
        ROOT.RDF.Distributed.DistributeCppCode(
            """
            bool check_number_less_than_four(int num){
                return num < 4;
            }  
            """
        )

    def _distribute_single_declare_check_filter_and_histo(self, connection):

        rdf = ROOT.RDataFrame(10, executor=connection)

        self._mydeclare_1(rdf)
        self._check_rdf_histos_1(rdf)

    def _distribute_multiple_declares_check_filter_and_histo(self, connection):

        rdf = ROOT.RDataFrame(10, executor=connection)

        self._mydeclare_1(rdf)
        self._mydeclare_2(rdf)
        self._check_rdf_histos_1(rdf)
        self._check_rdf_histos_2(rdf)

    def test_declares(self, payload):
        """
        Tests for the distribution of headers to the workers and their
        corresponding inclusion.
        """
        connection, _ = payload
        self._distribute_single_declare_check_filter_and_histo(connection)
        self._distribute_multiple_declares_check_filter_and_histo(connection)


if __name__ == "__main__":
    pytest.main(args=[__file__])
