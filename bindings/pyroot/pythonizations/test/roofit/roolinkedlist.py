import unittest

import ROOT


class TestRooLinkedList(unittest.TestCase):
    """
    Tests for the RooLinkedList.
    """

    def test_roolinkedlist_iteration(self):
        # test if we can correctly iterate over a RooLinkedList, also in
        # reverse.

        roolist = ROOT.RooLinkedList()
        pylist = []

        n_elements = 3

        for i in range(n_elements):
            obj = ROOT.TNamed(str(i), str(i))
            ROOT.SetOwnership(obj, False)
            roolist.Add(obj)
            pylist.append(obj)

        self.assertEqual(len(roolist), n_elements)

        for i, obj in enumerate(roolist):
            self.assertEqual(str(i), obj.GetName())


if __name__ == "__main__":
    unittest.main()
