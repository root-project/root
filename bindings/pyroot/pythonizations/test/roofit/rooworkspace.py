import unittest
import ROOT


class RooWorkspace_test(unittest.TestCase):
    """
    Test for the pythonizations of RooWorkspace.
    """

    # Setup
    def setUp(self):
        self.x = ROOT.RooRealVar("x", "x", 1.337, 0, 10)
        self.ws = ROOT.RooWorkspace("ws", "A workspace")

    # Tests
    def test_import(self):
        self.ws.Import(self.x)
        x = self.ws.var("x")
        self.assertEqual(x.GetName(), "x")
        self.assertEqual(x.getVal(), self.x.getVal())

    def test_import_with_arg(self):
        # Prepare workspace with variables and a PDF
        self.ws.Import(self.x)
        rename = ROOT.RooFit.RenameAllVariables("exp")
        exp = ROOT.RooExponential("exp", "exp", self.x, self.x)
        self.ws.Import(exp, rename)

        # Test that rename argument has worked
        x = self.ws.var("x_exp")
        self.assertEqual(x.getVal(), self.x.getVal())
        pdf = self.ws.pdf("exp")
        self.assertGreater(pdf.getVal(), 0)

    def test_import_argset(self):
        argSet = ROOT.RooArgSet(self.x)
        self.ws.Import(argSet)
        x = self.ws.arg("x")
        self.assertEqual(x.GetName(), "x")
        self.assertEqual(x.getVal(), self.x.getVal())

    def test_set_item_using_string(self):
        # Test to check if new variables are created
        self.ws["z"] = "[3]"
        self.assertEqual(self.ws["z"].GetName(), "z")
        self.assertEqual(self.ws["z"].getVal(), 3.0)

        # Test to check if new p.d.f.s are created
        self.ws["gauss"] = "Gaussian(x[0.0, 10.0], mu[5.0], sigma[2.0, 0.01, 10.0])"
        self.assertEqual(self.ws["gauss"].getX(), self.ws["x"])
        self.assertEqual(self.ws["gauss"].getMean(), self.ws["mu"])
        self.assertEqual(self.ws["gauss"].getSigma(), self.ws["sigma"])

    def test_set_item_using_dictionary(self):
        ws = ROOT.RooWorkspace()

        # Test to check if new variables are created
        ws["x"] = dict({"min": 0.0, "max": 10.0})
        self.assertEqual(ws["x"].getMax(), 10.0)
        self.assertEqual(ws["x"].getMin(), 0.0)

        # Test to check if new functions are created
        ws["m1"] = dict({"max": 5, "min": -5, "value": 0})
        ws["m2"] = dict({"max": 5, "min": -5, "value": 1})
        ws["mean"] = dict({"type": "sum", "summands": ["m1", "m2"]})
        self.assertEqual(ws["mean"].GetName(), "mean")
        self.assertEqual(ws["mean"].getVal(), 1.0)

        # Test to check if new p.d.f.s are created
        ws["sigma"] = dict({"value": 2, "min": 0.1, "max": 10.0})
        ws["gauss"] = dict({"mean": "mean", "sigma": "sigma", "type": "gaussian_dist", "x": "x"})
        self.assertEqual(ws["gauss"].getX(), ws["x"])
        self.assertEqual(ws["gauss"].getMean(), ws["mean"])
        self.assertEqual(ws["gauss"].getSigma(), ws["sigma"])


if __name__ == "__main__":
    unittest.main()
