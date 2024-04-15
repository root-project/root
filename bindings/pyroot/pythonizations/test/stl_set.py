import unittest
import ROOT


class STL_set(unittest.TestCase):
    """
    Tests for the pythonizations of std::set.
    """

    def test_set_char_data(self):
        """
        Test that a std::set of char behaves as a Python set.
        """

        elems = ['a', 'b', 'b', 'c']
        s = ROOT.std.set['char'](elems)
        self.assertEqual(set(s), set(elems))
        self.assertTrue(s)

        for entry in s:
            self.assertTrue(entry in set(elems))

    def test_set_types(self):
        """
        Instantiate std::set with different types.
        """

        for entry_type in ['int', 'float', 'double', 'char', 'const char*', 'std::string']:
            ROOT.std.set[entry_type]()

    def test_stl_set_boolean(self):
        """
        Test that the boolean conversion of a std::set works as expected.
        https://github.com/root-project/root/issues/14573
        """

        for entry_type in ['int', 'float', 'double']:
            s = ROOT.std.set[entry_type]()
            self.assertTrue(s.empty())
            self.assertFalse(bool(s))

            s.insert(1)
            self.assertFalse(s.empty())
            self.assertTrue(bool(s))

    def test_stl_set_tree(self):
        """
        Test that a TTree with a std::set branch behaves as expected.
        """

        tree = ROOT.TTree("tree", "Tree with std::vector")

        entries_to_fill = [
            set(),
            {1},
            {1, 2},
        ]

        # Create variables to store std::vector elements
        entry_root = ROOT.std.set(int)()

        # Create branches in the TTree
        tree.Branch("set", entry_root)

        for entry in entries_to_fill:
            entry_root.clear()
            for element in entry:
                entry_root.insert(element)
            tree.Fill()

        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            entry_python = entries_to_fill[i]
            self.assertEqual(entry_python, set(entry_root))
            self.assertEqual(bool(entry_python), bool(entry_root))
            self.assertEqual(len(entry_python), len(entry_root))


if __name__ == '__main__':
    unittest.main()
