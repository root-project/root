import unittest

import ROOT


class TH2UsingDecls(unittest.TestCase):
    """
    Test that the method pulled in via using decls from TH1 are accessible
    """

    # Tests
    def test_GetBinError(self):
        h = ROOT.TH2F("h", "h", 1, 1, 0, 1, 0, 1)
        for _ in range(4):
            h.Fill(1, 1, 1)
        self.assertEqual(h.GetBinErrorUp(1, 1), 2)
        self.assertEqual(h.GetBinErrorLow(1, 1), 2)


class TH2Operations(unittest.TestCase):
    """
    Test TH2D arithmetic operations
    """

    def setUp(self):
        self.h1 = ROOT.TH2D("h1", "h1", 2, 0, 2, 2, 0, 2)
        self.h2 = ROOT.TH2D("h2", "h2", 2, 0, 2, 2, 0, 2)

        self.h1.Fill(0.5, 0.5, 2.0)
        self.h2.Fill(0.5, 0.5, 3.0)

    def test_addition(self):
        hsum = self.h1 + self.h2
        self.assertAlmostEqual(hsum.GetBinContent(1, 1), 5.0)

    def test_subtraction(self):
        hdiff = self.h2 - self.h1
        self.assertAlmostEqual(hdiff.GetBinContent(1, 1), 1.0)

    def test_multiplication(self):
        hprod = self.h1 * self.h2
        self.assertAlmostEqual(hprod.GetBinContent(1, 1), 6.0)

    def test_division(self):
        hdiv = self.h2 / self.h1
        self.assertAlmostEqual(hdiv.GetBinContent(1, 1), 1.5)

    def test_scalar_multiplication_left(self):
        hscaled = 2.0 * self.h1
        self.assertAlmostEqual(hscaled.GetBinContent(1, 1), 4.0)

    def test_scalar_multiplication_right(self):
        hscaled = self.h1 * 2.0
        self.assertAlmostEqual(hscaled.GetBinContent(1, 1), 4.0)


if __name__ == "__main__":
    unittest.main()
