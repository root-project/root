import unittest
import os
import ROOT

class RBatchGeneratorMultipleFiles(unittest.TestCase):

    file_name1 = "first_half.root"
    file_name2 = "second_half.root"
    tree_name = "mytree"

    # Helpers
    def create_20_entries_file(self):
        df1 = ROOT.RDataFrame(20)\
            .Define("b1", "(int) rdfentry_")\
            .Define("b2", "(double) rdfentry_ * rdfentry_")\
            .Snapshot(self.tree_name, self.file_name1)
    

    def create_10_entries_file(self):
        df2 = ROOT.RDataFrame(10)\
            .Define("b1", "(int) rdfentry_ + 20")\
            .Define("b2", "(double) b1*b1")\
            .Snapshot(self.tree_name, self.file_name2)
        
        #print(df1.Describe())
        #print(df2.Describe())

        #df2.Display("",).Print()


    def teardown_file(self, file):
        os.remove(file)


    def test_foo(self):
        self.create_20_entries_file()

        ROOT.TMVA.Experimental.CreatePyTorchGenerators(
        tree_name=self.tree_name,
        file_name=self.file_name1,
        batch_size=10,
        chunk_size=30,
        target="b2",
        validation_split=0.3
        )

        self.teardown_file(self.file_name1)
        print("Tested")

if __name__ == 'main':
    unittest.main()