# \file
# \ingroup tutorial_dataframe
# \notebook -nodraw
#
# This example shows how to process a dataset where entries might be
# incomplete due to one or more missing branches in one or more of the files
# in the dataset. It shows usage of the FilterAvailable and DefaultValueFor
# RDataFrame functionalities to act upon the missing entries.
#
# \macro_code
# \macro_output
#
# \date September 2024
# \author Vincenzo Eduardo Padulano (CERN)
import array
import os

import ROOT


class DatasetContext:
    """A helper class to create the dataset for the tutorial below."""

    filenames = [
        "df036_missingBranches_py_file_1.root",
        "df036_missingBranches_py_file_2.root",
        "df036_missingBranches_py_file_3.root",
    ]
    treenames = ["tree_1", "tree_2", "tree_3"]
    nentries = 5

    def __init__(self):
        with ROOT.TFile(self.filenames[0], "RECREATE"):
            t = ROOT.TTree(self.treenames[0], self.treenames[0])
            x = array.array("i", [0])  # any array can also be a numpy array
            y = array.array("i", [0])
            t.Branch("x", x, "x/I")
            t.Branch("y", y, "y/I")

            for i in range(1, self.nentries + 1):
                x[0] = i
                y[0] = 2 * i
                t.Fill()

            t.Write()

        with ROOT.TFile(self.filenames[1], "RECREATE"):
            t = ROOT.TTree(self.treenames[1], self.treenames[1])
            y = array.array("i", [0])  # any array can also be a numpy array
            t.Branch("y", y, "y/I")

            for i in range(1, self.nentries + 1):
                y[0] = 3 * i
                t.Fill()

            t.Write()

        with ROOT.TFile(self.filenames[2], "RECREATE"):
            t = ROOT.TTree(self.treenames[2], self.treenames[2])
            x = array.array("i", [0])  # any array can also be a numpy array
            t.Branch("x", x, "x/I")

            for i in range(1, self.nentries + 1):
                x[0] = 4 * i
                t.Fill()

            t.Write()

    def __enter__(self):
        """Enable using the class as a context manager."""
        return self

    def __exit__(self, *_):
        """
        Enable using the class as a context manager. At the end of the context,
        remove the files created.
        """
        for filename in self.filenames:
            os.remove(filename)


def df036_missingBranches(dataset: DatasetContext):
    # The input dataset contains three files, with one TTree each.
    # The first contains branches (x, y), the second only branch y, the third
    # only branch x. The TChain will process the three files, encountering a
    # different missing branch when switching to the next tree
    chain = ROOT.TChain()
    for fname, tname in zip(dataset.filenames, dataset.treenames):
        chain.Add(fname + "?#" + tname)

    df = ROOT.RDataFrame(chain)

    default_value = ROOT.std.numeric_limits[int].min()

    # Example 1: provide a default value for all missing branches
    display_1 = (
        df.DefaultValueFor("x", default_value)
        .DefaultValueFor("y", default_value)
        .Display(columnList=("x", "y"), nRows=15)
    )

    # Example 2: provide a default value for branch y, but skip events where
    # branch x is missing
    display_2 = df.DefaultValueFor("y", default_value).FilterAvailable("x").Display(columnList=("x", "y"), nRows=15)

    # Example 3: only keep events where branch y is missing and display values for branch x
    display_3 = df.FilterMissing("y").Display(columnList=("x",), nRows=15)

    print("Example 1: provide a default value for all missing branches")
    display_1.Print()
    print("Example 2: provide a default value for branch y, but skip events where branch x is missing")
    display_2.Print()
    print("Example 3: only keep events where branch y is missing and display values for branch x")
    display_3.Print()


if __name__ == "__main__":
    with DatasetContext() as dataset:
        df036_missingBranches(dataset)
