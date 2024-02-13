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


if __name__ == '__main__':
    unittest.main()
