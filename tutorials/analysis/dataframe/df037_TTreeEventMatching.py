# \file
# \ingroup tutorial_dataframe
# \notebook -nodraw
#
# This example shows processing of a TTree-based dataset with horizontal
# concatenations (friends) and event matching (based on TTreeIndex). In case
# the current event being processed does not match one (or more) of the friend
# datasets, one can use the FilterAvailable and DefaultValueFor functionalities
# to act upon the situation.
#
# \macro_code
# \macro_output
#
# \date September 2024
# \author Vincenzo Eduardo Padulano (CERN)
import os
import ROOT
import numpy


class DatasetContext:
    """A helper class to create the dataset for the tutorial below."""

    main_file = "df037_TTreeEventMatching_py_main.root"
    aux_file_1 = "df037_TTreeEventMatching_py_aux_1.root"
    aux_file_2 = "df037_TTreeEventMatching_py_aux_2.root"
    main_tree_name = "events"
    aux_tree_name_1 = "auxdata_1"
    aux_tree_name_2 = "auxdata_2"

    def __init__(self):
        with ROOT.TFile(self.main_file, "RECREATE") as f:
            main_tree = ROOT.TTree(self.main_tree_name, self.main_tree_name)
            idx = numpy.array([0], dtype=int)
            x = numpy.array([0], dtype=int)
            main_tree.Branch("idx", idx, "idx/I")
            main_tree.Branch("x", x, "x/I")

            idx[0] = 1
            x[0] = 1
            main_tree.Fill()
            idx[0] = 2
            x[0] = 2
            main_tree.Fill()
            idx[0] = 3
            x[0] = 3
            main_tree.Fill()

            main_tree.Write()

        # The first auxiliary file has matching indices 1 and 2, but not 3
        with ROOT.TFile(self.aux_file_1, "RECREATE") as f:
            aux_tree_1 = ROOT.TTree(self.aux_tree_name_1, self.aux_tree_name_1)
            idx = numpy.array([0], dtype=int)
            y = numpy.array([0], dtype=int)
            aux_tree_1.Branch("idx", idx, "idx/I")
            aux_tree_1.Branch("y", y, "y/I")

            idx[0] = 1
            y[0] = 4
            aux_tree_1.Fill()
            idx[0] = 2
            y[0] = 5
            aux_tree_1.Fill()

            aux_tree_1.Write()

        # The second auxiliary file has matching indices 1 and 3, but not 2
        with ROOT.TFile(self.aux_file_2, "RECREATE") as f:
            aux_tree_2 = ROOT.TTree(self.aux_tree_name_2, self.aux_tree_name_2)
            idx = numpy.array([0], dtype=int)
            z = numpy.array([0], dtype=int)
            aux_tree_2.Branch("idx", idx, "idx/I")
            aux_tree_2.Branch("z", z, "z/I")

            idx[0] = 1
            z[0] = 6
            aux_tree_2.Fill()
            idx[0] = 3
            z[0] = 7
            aux_tree_2.Fill()

            aux_tree_2.Write()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        os.remove(self.main_file)
        os.remove(self.aux_file_1)
        os.remove(self.aux_file_2)


def df037_TTreeEventMatching(dataset: DatasetContext):
    # The input dataset has one main TTree and two auxiliary. The 'idx' branch
    # is used as the index to match events between the trees.
    # - The main tree has 3 entries, with 'idx' values(1, 2, 3).
    # - The first auxiliary tree has 2 entries, with 'idx' values(1, 2).
    # - The second auxiliary tree has 2 entries, with 'idx' values(1, 3).
    # The two auxiliary trees are concatenated horizontally with the main one.
    main_chain = ROOT.TChain(dataset.main_tree_name)
    main_chain.Add(dataset.main_file)

    aux_chain_1 = ROOT.TChain(dataset.aux_tree_name_1)
    aux_chain_1.Add(dataset.aux_file_1)
    aux_chain_1.BuildIndex("idx")

    aux_chain_2 = ROOT.TChain(dataset.aux_tree_name_2)
    aux_chain_2.Add(dataset.aux_file_2)
    aux_chain_2.BuildIndex("idx")

    main_chain.AddFriend(aux_chain_1)
    main_chain.AddFriend(aux_chain_2)

    # Create an RDataFrame to process the input dataset. The DefaultValueFor and
    # FilterAvailable functionalities can be used to decide what to do for
    # the events that do not match entirely according to the index column 'idx'
    df = ROOT.RDataFrame(main_chain)

    aux_tree_1_colidx = dataset.aux_tree_name_1 + ".idx"
    aux_tree_1_coly = dataset.aux_tree_name_1 + ".y"
    aux_tree_2_colidx = dataset.aux_tree_name_2 + ".idx"
    aux_tree_2_colz = dataset.aux_tree_name_2 + ".z"

    default_value = ROOT.std.numeric_limits[int].min()

    # Example 1: provide default values for all columns in case there was no
    # match
    display_1 = (
        df.DefaultValueFor(aux_tree_1_colidx, default_value)
        .DefaultValueFor(aux_tree_1_coly, default_value)
        .DefaultValueFor(aux_tree_2_colidx, default_value)
        .DefaultValueFor(aux_tree_2_colz, default_value)
        .Display(
            ("idx", aux_tree_1_colidx, aux_tree_2_colidx, "x", aux_tree_1_coly, aux_tree_2_colz))
    )

    # Example 2: skip the entire entry when there was no match for a column
    # in the first auxiliary tree, but keep the entries when there is no match
    # in the second auxiliary tree and provide a default value for those
    display_2 = (
        df.DefaultValueFor(aux_tree_2_colidx, default_value)
        .DefaultValueFor(aux_tree_2_colz, default_value)
        .FilterAvailable(aux_tree_1_coly)
        .Display(
                ("idx", aux_tree_1_colidx, aux_tree_2_colidx, "x", aux_tree_1_coly, aux_tree_2_colz))
    )

   # Example 3: Keep entries from the main tree for which there is no
   # corresponding match in entries of the first auxiliary tree
    display_3 = df.FilterMissing(aux_tree_1_colidx).Display(("idx", "x"))

    print("Example 1: provide default values for all columns")
    display_1.Print()
    print("Example 2: always skip the entry when there is no match")
    display_2.Print()
    print("Example 3: keep entries from the main tree for which there is no match in the auxiliary tree")
    display_3.Print()


if __name__ == "__main__":
    with DatasetContext() as dataset:
        df037_TTreeEventMatching(dataset)
