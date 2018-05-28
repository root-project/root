import unittest
import ROOT

dtypes = [
        "int", "unsigned int", "long", "unsigned long", "float", "double"
    ]

rvecs = {}

class VecOps(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
       for t in dtypes:
          v = ROOT.ROOT.VecOps.RVec(t)(5)
          for i in range(len(v)): v[i] = i
          rvecs[t] = v

    # Tests
    def test_Loop(self):
        for t, v in rvecs.items():
           print("Content of the RVec: {}".format([x for x in v]))

if __name__ == '__main__':
    unittest.main()
