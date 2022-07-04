import os

import pytest

import ROOT
from DistRDF.Backends import Dask


TREENAMES = [f"distrdf_check_friend_trees_alignment_dask_tree_{i}.root" for i in range(1, 7)]
FILENAMES = [f"distrdf_check_friend_trees_alignment_dask_file_{i}.root" for i in range(1, 7)]


def create_dataset():
    df = ROOT.RDataFrame(60).Define("x", "rdfentry_")

    range_limits = list(range(0, 61, 10))
    opts = ROOT.RDF.RSnapshotOptions()
    opts.fAutoFlush = 1
    for idx, (begin, end) in enumerate(zip(range_limits, range_limits[1:])):
        df.Range(begin, end).Snapshot(TREENAMES[idx], FILENAMES[idx], ["x"], opts)


def create_chain():

    main = ROOT.TChain()
    for i in range(3):
        main.Add(f"{FILENAMES[i]}?#{TREENAMES[i]}")

    friend = ROOT.TChain()
    for i in range(3, 6):
        friend.Add(f"{FILENAMES[i]}?#{TREENAMES[i]}")

    main.AddFriend(friend, "friend")

    return main


@pytest.fixture(scope="class")
def setup_testfriendtreesalignment(request):
    """
    Set up test environment for this module. Currently this includes:

    - Write a set of trees usable by the tests in this class
    - Remove the files at the end of the tests.
    """

    create_dataset()
    yield
    for name in FILENAMES:
        os.remove(name)


@pytest.mark.usefixtures("setup_testfriendtreesalignment")
class TestDaskFriendTreesAlignment:
    """Check that distrdf processes the correct entries from friend trees"""

    def test_sum_friend_column(self, connection):
        """
        Define one main chain and one friend chain, with different entry values.
        The main chain has one column with values in range [0, 30). The friend
        chain has one column with values in range [30, 60). Computing the sum of
        values in the friend column tells whether the right entries were read
        from the friend chain, hence ensuring the correct alignment.
        """

        chain = create_chain()

        for nparts in range(1, 11):
            df = Dask.RDataFrame(chain, daskclient=connection, npartitions=nparts)
            s1 = df.Sum("x")
            s2 = df.Sum("friend.x")
            s1val = s1.GetValue()
            s2val = s2.GetValue()
            assert s1val == 435, f"{s1val} != 435"  # sum(range(0, 30))
            assert s2val == 1335, f"{s2val} != 1335"  # sum(range(30, 60))


if __name__ == "__main__":
    pytest.main(args=[__file__])
