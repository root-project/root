import unittest

import ROOT


class TVectorTLen(unittest.TestCase):
    """
    Test for the pythonization that allows to get the size of a
    TVectorT by calling `len` on it.
    """

    num_elems = 3

    # Tests
    def test_len(self):
        v = ROOT.TVectorT[float](self.num_elems)
        self.assertEqual(len(v), self.num_elems)


class TVectorTGetItem(unittest.TestCase):
    """
    Test for the pythonization that allows to: (i) get an item of a
    TVectorT with boundary check for the index and (ii) iterate over
    a TVectorT.
    """

    num_elems = 3

    # Tests
    def test_boundary_check(self):
        v = ROOT.TVectorT[float](self.num_elems)

        # In range
        self.assertEqual(v[0], v[0])

        # Out of range
        with self.assertRaises(IndexError):
            v[-1]

        # Out of range
        with self.assertRaises(IndexError):
            v[self.num_elems]

    def test_iterable(self):
        v = ROOT.TVectorT[float](self.num_elems)
        val = 1

        for i in range(self.num_elems):
            v[i] = val

        self.assertEqual(list(v), [ val for _ in range(self.num_elems) ])


if __name__ == '__main__':
    unittest.main()
