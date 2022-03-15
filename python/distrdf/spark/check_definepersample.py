import os

import pytest

import DistRDF
from DistRDF.Backends import Spark

import ROOT


def write_tree(treename, filename):
    df = ROOT.RDataFrame(10)
    df.Define("x", "rdfentry_").Snapshot(treename, filename)


@pytest.fixture(scope="class")
def setup_testdefinepersample(request):
    """
    Set up test environment for this class. Currently this includes:

    - Write a set of trees usable by the tests in this class
    - Initialize a Dask client for the tests in this class. This uses a
        `LocalCluster` object that spawns 2 single-threaded Python processes.
    """

    request.cls.samples = ["sample1", "sample2", "sample3"]
    request.cls.filenames = [sample + ".root" for sample in request.cls.samples]
    request.cls.maintreename = "Events"
    for filename in request.cls.filenames:
        write_tree(request.cls.maintreename, filename)
    yield
    for name in request.cls.filenames:
        os.remove(name)


@pytest.mark.usefixtures("setup_testdefinepersample")
class TestDefinePerSample:
    """Check the working of merge operations in the reducer function."""

    def test_definepersample_simple(self, connection):
        """
        Test DefinePerSample operation on three samples using a predefined
        string of operations.
        """

        df = Spark.RDataFrame(self.maintreename, self.filenames, sparkcontext=connection)

        # Associate a number to each sample
        definepersample_code = """
        if(rdfsampleinfo_.Contains(\"{}\")) return 1;
        else if (rdfsampleinfo_.Contains(\"{}\")) return 2;
        else if (rdfsampleinfo_.Contains(\"{}\")) return 3;
        else return 0;
        """.format(*self.samples)

        df1 = df.DefinePerSample("sampleid", definepersample_code)

        # Filter by the sample number. Each filtered dataframe should contain
        # 10 entries, equal to the number of entries per sample
        samplescounts = [df1.Filter("sampleid == {}".format(id)).Count() for id in [1, 2, 3]]

        for count in samplescounts:
            assert count.GetValue() == 10

    def test_definepersample_withinitialization(self, connection):
        """
        Test DefinePerSample operation on three samples using C++ functions
        declared to the ROOT interpreter.
        """

        # Write initialization code that will be run in the workers to make the
        # needed functions available
        def declare_definepersample_code():
            ROOT.gInterpreter.Declare(
                '''
            #ifndef distrdf_test_definepersample_withinitialization
            #define distrdf_test_definepersample_withinitialization
            float sample1_weight(){
                return 1.0f;
            }

            float sample2_weight(){
                return 2.0f;
            }

            float sample3_weight(){
                return 3.0f;
            }

            float samples_weights(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
                if (id.Contains("sample1")){
                    return sample1_weight();
                } else if (id.Contains("sample2")){
                    return sample2_weight();
                } else if (id.Contains("sample3")){
                    return sample3_weight();
                }
                return -999.0f;
            }

            std::string samples_names(unsigned int slot, const ROOT::RDF::RSampleInfo &id){
                return id.AsString();
            }
            #endif // distrdf_test_definepersample_withinitialization
            ''')

        DistRDF.initialize(declare_definepersample_code)
        df = Spark.RDataFrame(self.maintreename, self.filenames, sparkcontext=connection)
        df1 = df.DefinePerSample("sample_weight", "samples_weights(rdfslot_, rdfsampleinfo_)")\
                .DefinePerSample("sample_name", "samples_names(rdfslot_, rdfsampleinfo_)")

        # Filter by the two defined columns per sample: a weight and the sample string representation
        # Each filtered dataset should have 10 entries, equal to the number of entries per sample
        weightsandnames = [("1.0f", "sample1.root/Events"), ("2.0f", "sample2.root/Events"),
                           ("3.0f", "sample3.root/Events")]
        samplescounts = [
            df1.Filter("sample_weight == {} && sample_name == \"{}\"".format(weight, name)).Count()
            for (weight, name) in weightsandnames]

        for count in samplescounts:
            assert count.GetValue() == 10


if __name__ == "__main__":
    pytest.main(args=[__file__])
