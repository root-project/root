import os
import unittest

import ROOT

import DistRDF
from DistRDF.Backends import Dask

from dask.distributed import Client, LocalCluster

def write_tree(treename, filename):
    df = ROOT.RDataFrame(10)
    df.Define("x", "rdfentry_").Snapshot(treename, filename)


class DefinePerSampleTest(unittest.TestCase):
    """Check the working of merge operations in the reducer function."""

    @classmethod
    def setUpClass(cls):
        """
        Set up test environment for this class. Currently this includes:

        - Write a set of trees usable by the tests in this class
        - Initialize a Dask client for the tests in this class. This uses a
          `LocalCluster` object that spawns 2 single-threaded Python processes.
        """

        cls.samples = ["sample1", "sample2", "sample3"]
        cls.filenames = [sample + ".root" for sample in cls.samples]
        cls.maintreename = "Events"
        for filename in cls.filenames:
            write_tree(cls.maintreename, filename)

        cls.client = Client(LocalCluster(n_workers=2, threads_per_worker=1, processes=True))

    @classmethod
    def tearDownClass(cls):
        """Reset test environment."""

        cls.client.shutdown()
        cls.client.close()

        for name in cls.filenames:
            os.remove(name)

    def test_definepersample_simple(self):
        """
        Test DefinePerSample operation on three samples using a predefined
        string of operations.
        """

        df = Dask.RDataFrame(self.maintreename, self.filenames, daskclient=self.client)

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
            self.assertEqual(count.GetValue(), 10)

    def test_definepersample_withinitialization(self):
        """
        Test DefinePerSample operation on three samples using C++ functions
        declared to the ROOT interpreter.
        """

        # Write initialization code that will be run in the workers to make the
        # needed functions available
        def declare_definepersample_code():
            ROOT.gInterpreter.Declare(
                '''

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

            ''')

        DistRDF.initialize(declare_definepersample_code)
        df = Dask.RDataFrame(self.maintreename, self.filenames, daskclient=self.client)
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
            self.assertEqual(count.GetValue(), 10)


if __name__ == "__main__":
    unittest.main(argv=[__file__])
