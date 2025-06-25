import array
import random
import unittest

import ROOT


# Helper function to assert that the elements of an array object and a std::vector proxy are equal
def assertVec(vec, arr):
    # cppyy automatically casts random integers to unicode characters,
    # so do the same in python so the validation doesn't fail
    if isinstance(arr, array.array) and arr.typecode in ("b", "B"):
        arr = [chr(b) for b in arr]

    tc = unittest.TestCase()
    # first check lengths match
    tc.assertEqual(len(vec), len(arr), f"Length mismatch: std::vector is {len(vec)}, array is {len(arr)}")

    tc.assertSequenceEqual(vec, arr, msg="std::vector and array differ, iadd failed")


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

    def test_stl_vector_iadd(self):
        import array
        import random

        """
        Test that the __iadd__ pythonization of std::vector works as expected.
        This call dispatches to std::insert
        https://github.com/root-project/root/issues/18768
        """

        # we go over all possible numeric PODs
        # https://docs.python.org/3/library/array.html
        entry_type = [
            "char",
            "unsigned char",
            "short",
            "unsigned short",
            "int",
            "unsigned int",
            "long",
            "unsigned long",
            "long long",
            "float",
            "double",
        ]
        array_type = ["b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d"]

        typemap = zip(entry_type, array_type)
        n = 5
        for dtype in typemap:
            vec = ROOT.std.vector[dtype[0]]()
            self.assertTrue(vec.empty())
            li = [random.randint(1, 100) for _ in range(n)]
            arr = array.array(dtype[1], li)
            vec += arr
            self.assertFalse(vec.empty())
            assertVec(vec, arr)
            vec.pop_back()
            arr = arr[:-1]
            assertVec(vec, arr)

    def test_stl_vector_iadd_2D(self):
        """
        Test that the __iadd__ pythonization of std::vector works as expected in 2D
        """
        initial = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        vec2d = ROOT.std.vector[ROOT.std.vector[int]](initial)
        self.assertEqual(
            len(vec2d), len(initial), f"Initial 2D vector row count wrong ({len(vec2d)} vs {len(initial)})"
        )

        # verify rows before iadd
        for idx, (subvec, sublist) in enumerate(zip(vec2d, initial)):
            with self.subTest(row=idx, phase="before"):
                assertVec(subvec, sublist)

        vec2d += initial
        expected = initial + initial

        self.assertEqual(
            len(vec2d), len(expected), f"2D vector row count after iadd wrong ({len(vec2d)} vs {len(expected)})"
        )

        for idx, (subvec, sublist) in enumerate(zip(vec2d, expected)):
            with self.subTest(row=idx, phase="after"):
                assertVec(subvec, sublist)

    def test_tree_with_containers(self):
        """
        Test that the boolean conversion of a std::vector works as expected inside a TTree.
        Also checks that the contents are correctly filled and read back.
        https://github.com/root-project/root/issues/14573
        """

        # Create a TTree
        tree = ROOT.TTree("tree", "Tree with std::vector")

        # list of random arrays with lengths between 0 and 5 (0 is always included)
        entries_to_fill = [array.array("f", [random.uniform(10, 20) for _ in range(n % 5)]) for n in range(100)]

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
            entry_array = entries_to_fill[i]

            self.assertEqual(len(entry_array), len(entry_root))

            self.assertEqual(bool(list(entry_root)), bool(entry_root))  # numpy arrays cannot be converted to bool

            for entry_array_i, entry_root_i in zip(entry_array, entry_root):
                self.assertEqual(entry_array_i, entry_root_i)


if __name__ == '__main__':
    unittest.main()
