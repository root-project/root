import pytest

import ROOT
from DistRDF.Backends import Dask


TREENAMES = [
    f"distrdf_roottest_check_friend_trees_alignment_{i}" for i in range(1, 7)]
FILENAMES = [
    f"../data/ttree/distrdf_roottest_check_friend_trees_alignment_{i}.root" for i in range(1, 7)]


def create_chain():

    main = ROOT.TChain()
    for i in range(3):
        main.Add(f"{FILENAMES[i]}?#{TREENAMES[i]}")

    friend = ROOT.TChain()
    for i in range(3, 6):
        friend.Add(f"{FILENAMES[i]}?#{TREENAMES[i]}")

    # The friend will be owned by the main TChain
    ROOT.SetOwnership(friend, False)

    main.AddFriend(friend, "friend")

    return main


class TestDaskFriendTreesAlignment:
    """Check that distrdf processes the correct entries from friend trees"""

    @pytest.mark.parametrize("nparts", range(1, 11))
    def test_sum_friend_column(self, payload, nparts):
        """
        Define one main chain and one friend chain, with different entry values.
        The main chain has one column with values in range [0, 30). The friend
        chain has one column with values in range [30, 60). Computing the sum of
        values in the friend column tells whether the right entries were read
        from the friend chain, hence ensuring the correct alignment.
        """

        chain = create_chain()

        connection, _ = payload
        df = ROOT.RDataFrame(chain, executor=connection, npartitions=nparts)

        s1 = df.Sum("x")
        s2 = df.Sum("friend.x")
        s1val = s1.GetValue()
        s2val = s2.GetValue()
        assert s1val == 435, f"{s1val} != 435"  # sum(range(0, 30))
        assert s2val == 1335, f"{s2val} != 1335"  # sum(range(30, 60))


if __name__ == "__main__":
    pytest.main(args=[__file__])
