"""
Functions to create input datasets for distributed RDataFrame.
"""
import ROOT
from array import array
import subprocess
import os
import numpy

def create_check_backend():
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fAutoFlush = 10
    df = ROOT.RDataFrame(100).Define("x", "1")
    treenames = [f"tree_{i}" for i in range(3)]
    filenames = [
        f"distrdf_roottest_check_backend_{i}.root" for i in range(3)]
    for treename, filename in zip(treenames, filenames):
        df.Snapshot(treename, filename, ["x"], opts)


def create_emtpy_ttree():
    with ROOT.TFile("empty.root", "recreate") as f:
        tree = ROOT.TTree("empty", "empty")
        f.WriteObject(tree, tree.GetName())


def create_definepersample():
    samples = ["sample1", "sample2", "sample3"]
    filenames = [
        f"distrdf_roottest_definepersample_{sample}.root" for sample in samples]
    maintreename = "Events"
    for filename in filenames:
        df = ROOT.RDataFrame(10)
        df.Define("x", "rdfentry_").Snapshot(maintreename, filename)


def create_friend_trees_alignment():
    treenames = [
        f"distrdf_roottest_check_friend_trees_alignment_{i}" for i in range(1, 7)]
    filenames = [
        f"distrdf_roottest_check_friend_trees_alignment_{i}.root" for i in range(1, 7)]

    df = ROOT.RDataFrame(60).Define("x", "rdfentry_")

    range_limits = list(range(0, 61, 10))
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fAutoFlush = 1
    for idx, (begin, end) in enumerate(zip(range_limits, range_limits[1:])):
        df.Range(begin, end).Snapshot(
            treenames[idx], filenames[idx], ["x"], opts)


def create_friend_trees():
    def create_tree(treename, filename, gaus_mean):

        with ROOT.TFile(filename, "recreate") as f:
            t = ROOT.TTree(treename, treename)

            x = array("f", [0])
            t.Branch("x", x, "x/F")

            r = ROOT.TRandom()
            for _ in range(10000):
                x[0] = r.Gaus(gaus_mean, 1)
                t.Fill()
            f.WriteObject(t, t.GetName())

    main_treename = "T"
    friend_treename = "TF"
    main_mean = 10
    friend_mean = 20
    main_filename = "distrdf_roottest_check_friend_trees_main.root"
    friend_filename = "distrdf_roottest_check_friend_trees_friend.root"

    create_tree(main_treename, main_filename, main_mean)
    create_tree(friend_treename, friend_filename, friend_mean)
    
    #7584
    rn1 = "rn1.root"
    rn2 = "rn2.root"
    friendsfilename = "distrdf_roottest_check_friend_trees_7584.root"

    df_1 = ROOT.RDataFrame(10000)
    df_2 = ROOT.RDataFrame(10000)

    df_1 = df_1.Define("rnd", "gRandom->Gaus(10)")
    df_2 = df_2.Define("rnd", "gRandom->Gaus(20)")

    df_1.Snapshot("randomNumbers", rn1)
    df_2.Snapshot("randomNumbersBis", rn2)

    # Put the two trees together in a common file
    subprocess.run("hadd -f {} {} {}".format(friendsfilename, rn1, rn2),
                shell=True, check=True)
    os.remove(rn1)
    os.remove(rn2)

def create_reducer_merge():
    # Create dataset with fixed series of entries
    treename = "tree"
    filename = "distrdf_roottest_check_reducer_merge_1.root"
    ROOT.RDataFrame(100).Define("v", "static_cast<double>(rdfentry_)").Snapshot(treename, filename)

def create_rungraphs():
    # Create a test file for processing
    treename = "tree"
    filename = "distrdf_roottest_check_rungraphs.root"
    nentries = 10000
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fAutoFlush = 5000
    ROOT.RDataFrame(nentries).Define("b1", "42")\
                                .Define("b2", "42")\
                                .Define("b3", "42")\
                                .Snapshot(treename, filename, ["b1", "b2", "b3"], opts)


def create_cloned_actions():
    # 20 cluster boundaries of variable distance
    clusters = [
        66, 976, 1542, 1630, 2477, 3566, 4425, 4980, 5109, 5381, 5863, 6533, 6590,
        6906, 8312, 8361, 8900, 8952, 9144, 9676
    ]
    datasetname = "Events"
    filename = "distrdf_roottest_check_cloned_actions_asnumpy.root"

    with ROOT.TDirectory.TContext(), ROOT.TFile(filename, "recreate") as f:
        t = ROOT.TTree(datasetname, datasetname)

        event = numpy.array([0], dtype=numpy.int64)
        t.Branch("event", event, "event/L")

        for i in range(10000):
            event[0] = i
            # Flush a cluster of entries at the defined cluster boundaries
            if i in clusters:
                t.FlushBaskets()
            t.Fill()
        f.WriteObject(t, t.GetName())
        
def create_fromspec():
    # Create a test file for processing
    treename = "tree"
    filename = "distrdf_roottest_check_fromspec.root"
    nentries = 1000
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fAutoFlush = 5000
    ROOT.RDataFrame(nentries).Define("b1", "50").Snapshot(treename, filename, ["b1"], opts)
