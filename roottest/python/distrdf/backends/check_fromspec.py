import DistRDF
import pytest
import ROOT


class TestFromSpec:
    
    def test_fromspec_different_trees(self, payload):
        """
        Test usage of FromSpec function when each sample has different trees
        """

        connection, _ = payload
                
        jsonfile = "../data/ttree/spec_differenttrees.json"
        df = ROOT.RDF.Experimental.FromSpec(jsonfile, executor = connection)
        
        df_checkfilt = df.FilterAvailable("nElectron").Filter("nElectron > 2")
        
        df_new = df.DefinePerSample("lum", 'rdfsampleinfo_.GetD("lum")')
        df_filtered = df_new.Filter("lum == 100.")
        df_filtered_two = df_new.Filter("lum == 200.")
        
        df_local = ROOT.RDF.Experimental.FromSpec(jsonfile)
        df_new_local = df_local.DefinePerSample("lum", 'rdfsampleinfo_.GetD("lum")')
        df_filtered_local = df_new_local.Filter("lum == 100.")
        df_filtered_two_local = df_new_local.Filter("lum == 200.")
        
        assert df_checkfilt.Count().GetValue() == 1683
        assert df_filtered.Count().GetValue() == 11000
        assert df_filtered.Count().GetValue() == df_filtered_local.Count().GetValue()

        assert df_filtered_two.Count().GetValue() == 13020
        assert df_filtered_two.Count().GetValue() == df_filtered_two_local.Count().GetValue()

    def test_fromspec_files_multiple_trees(self, payload):
        """
        Test usage of FromSpec function when some samples have multiple trees
        """
        connection, _ = payload
        
        jsonfile_two = "../data/ttree/spec.json"
        
        rdf = ROOT.RDF.Experimental.FromSpec(jsonfile_two, executor = connection)
        
        rdf_filt = rdf.FilterAvailable("b1").Filter("b1 > 42")
        
        nentries = 3000

        assert rdf_filt.Count().GetValue() == nentries
        assert rdf_filt.Mean("b1").GetValue() == 50

        rdf_new = rdf.DefinePerSample("lum", 'rdfsampleinfo_.GetD("lum")')
        
        rdf_filtered = rdf_new.Filter("lum == 200.")
        rdf_filtered_two = rdf_new.Filter("lum == 100.")
        rdf_filtered_three = rdf_new.Filter("lum == 5.") 
        rdf_filtered_four = rdf_new.Filter("lum == 10.")
        
        assert rdf_filtered.Count().GetValue() == 30000
        assert rdf_filtered_two.Count().GetValue() == 3000        
        assert rdf_filtered_three.Count().GetValue() == 2
        assert rdf_filtered_four.Count().GetValue() == 3
    
    def test_fromspec_with_friends(self, payload):
        """
        Test usage of FromSpec function when friends trees are added 
        """
        
        connection, _ = payload
    
        jsonfile_three = "../data/ttree/spec_withfriends.json"
        rdf_friends = ROOT.RDF.Experimental.FromSpec(jsonfile_three, executor = connection)
        
        rdf_friends_new = rdf_friends.DefinePerSample("lumi", 'rdfsampleinfo_.GetD("lumi")')
        
        rdf_friends_filtered = rdf_friends_new.Filter("lumi == 1.")
        rdf_friends_filtered_two = rdf_friends_new.Filter("lumi == 0.5")
        
        rdf_friends_values = rdf_friends_filtered_two.Filter("friendTree.z > 103")
        rdf_friends_values_two = rdf_friends.Filter("friendChain1.z > 102")
        
        assert rdf_friends_filtered.Count().GetValue() == 4
        assert rdf_friends_filtered_two.Count().GetValue() == 1
        assert rdf_friends_values.Count().GetValue() == 1
        assert rdf_friends_values_two.Count().GetValue() == 2

if __name__ == "__main__":
    pytest.main(args=[__file__])
    