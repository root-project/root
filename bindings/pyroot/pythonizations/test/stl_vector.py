import unittest
import ROOT
import random
import numpy as np


class STL_vector(unittest.TestCase):
    """
    Tests for the pythonizations of std::vector. 
    """

    def test_vec_char_data(self):
        """
        Test that calling std::vector<char>::data() returns a Python string
        that contains the characters of the vector and no exception is raised.
        Check also that the iteration over the vector runs normally (#9632).
        """

        elems = ['a', 'b', 'c']
        v = ROOT.std.vector['char'](elems)
        self.assertEqual(v.data(), ''.join(elems))

        for elem in v:
            self.assertEqual(elem, elems.pop(0))

    def test_vec_const_char_p(self):
        """
        Test that creating a std::vector<const char*> does not raise any
        exception (#11581).
        """
        ROOT.std.vector['const char*']()

    def test_stl_vector_boolean(self):
        """
        Test that the boolean conversion of a std::vector works as expected.
        https://github.com/root-project/root/issues/14573
        """
        for entry_type in ['int', 'float', 'double']:
            vector = ROOT.std.vector[entry_type]()
            self.assertTrue(vector.empty())
            self.assertFalse(bool(vector))

            vector.push_back(1)
            self.assertFalse(vector.empty())
            self.assertTrue(bool(vector))

    def test_tree_with_containers(self):
        """
        Test that the boolean conversion of a std::vector works as expected inside a TTree.
        Also checks that the contents are correctly filled and read back.
        https://github.com/root-project/root/issues/14573
        """

        # Create a TTree
        tree = ROOT.TTree("tree", "Tree with std::vector")

        # list of random arrays with lengths between 0 and 5 (0 is always included)
        entries_to_fill = [
            np.array([random.uniform(10, 20) for _ in range(n % 5)]) for n in range(100)
        ]

        # Create variables to store std::vector elements
        entry_root = ROOT.std.vector(float)()

        # Create branches in the TTree
        tree.Branch("vector", entry_root)

        # Fill the TTree with 100 entries
        for entry in entries_to_fill:
            entry_root.clear()

            for element in entry:
                entry_root.push_back(element)

            tree.Fill()

        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            entry_numpy = entries_to_fill[i]
            entry_python_list = list(entry_root)

            self.assertEqual(len(entry_numpy), len(entry_root))
            self.assertEqual(bool(entry_python_list), bool(entry_root))  # numpy arrays cannot be converted to bool
            np.testing.assert_allclose(entry_numpy, np.array(entry_root), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
