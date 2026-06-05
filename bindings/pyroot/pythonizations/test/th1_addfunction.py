import unittest

import ROOT

class TH1AddFunction(unittest.TestCase):
    def test_adding_a_function_does_not_segfault(self):
        """
        This test verifies that box is only freed once
        """
        h = ROOT.TH1F("h", "h", 10, 0, 1)
        box = ROOT.TBox(0.1, 0.1, 0.9, 0.9)
        h.GetListOfFunctions().Add(box)

if __name__ == "__main__":
    unittest.main()
