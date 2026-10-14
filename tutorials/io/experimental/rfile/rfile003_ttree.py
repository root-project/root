# \file
# \ingroup ROOT7 tutorial_io
# Demonstrate how to read a TTree with RFile.
#
# Note that, due to how TTree handles lifetime and its relationship with TDirectory, writing TTree through RFile
# is currently unsupported.
#
# \macro_code
# \macro_output
#
# \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
# is welcome!
#
# \date 2026-10-12
# \author Giacomo Parolini <giacomo.parolini@cern.ch>

import array
import os

import ROOT


def read_ttree_from_rfile(fileName):
    # Open an existing ROOT file for reading (will throw an exception if `fileName` cannot be read).
    # IMPORTANT! This file must remain open while reading data from the tree.
    with ROOT.Experimental.RFile.Open(fileName) as file:
        # Get back the TTree from the file
        tree = file.Get("tree")
        floatVal = array.array("f", [0])
        tree.SetBranchAddress("floatBranch", floatVal)

        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            print(f"value[{i}] = {floatVal[0]}")


fileName = "rfile_ttree_py.root"
try:
    # Note that TTree cannot be written to RFile, so we use RDataFrame to write it instead.
    ROOT.RDataFrame(10).Define("floatBranch", "static_cast<float>(rdfentry_)").Snapshot("tree", fileName)

    read_ttree_from_rfile(fileName)
    os.remove(fileName)
except FileNotFoundError:
    pass
