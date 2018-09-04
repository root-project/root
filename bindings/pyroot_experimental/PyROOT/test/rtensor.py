import unittest
import ROOT
import numpy as np

RTensor = ROOT.TMVA.Experimental.RTensor


class ArrayInterface(unittest.TestCase):
    dtypes = [
        "int", "unsigned int", "long", "unsigned long", "float", "double"
    ]

    def test_getitem(self):
        for dtype in self.dtypes:
            data = ROOT.std.vector(dtype)((0, 1, 2, 3, 4, 5))
            shape = ROOT.std.vector("size_t")((2, 3))
            x = RTensor(dtype)(data.data(), shape)
            count = 0
            for i in range(2):
                for j in range(3):
                    self.assertEqual(x[i, j], count)
                    count += 1

    def test_setitem(self):
        for dtype in self.dtypes:
            shape = ROOT.std.vector("size_t")((2, 3))
            x = RTensor(dtype)(shape)
            count = 0
            for i in range(2):
                for j in range(3):
                    x[i, j] = count
                    count += 1
            count = 0
            for i in range(2):
                for j in range(3):
                    self.assertEqual(x[i, j], count)
                    count += 1

    def test_dim1(self):
        shape = ROOT.std.vector("size_t")()
        shape.push_back(2)
        x = RTensor("float")(shape)
        x[0] = 1
        x[1] = 2
        self.assertEqual(x[0], 1)
        self.assertEqual(x[1], 2)


if __name__ == '__main__':
    unittest.main()
