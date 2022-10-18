import unittest
import ROOT


class RooSimultaneous_test(unittest.TestCase):
    """
    Test for the pythonizations of RooSimultaneous.
    """

    # Tests
    def test_construction_from_dict(self):

        x = ROOT.RooRealVar("x", "x", -10, 10)

        # Define model for each sample
        model = ROOT.RooGaussian("model", "model", x, ROOT.RooFit.RooConst(1.0), ROOT.RooFit.RooConst(1.0))
        model_ctl = ROOT.RooGaussian("model_ctl", "model_ctl", x, ROOT.RooFit.RooConst(-1.0), ROOT.RooFit.RooConst(1.0))

        # Define category to distinguish physics and control samples events
        sample = ROOT.RooCategory("sample", "sample")
        sample.defineType("physics")
        sample.defineType("control")

        # Construct the RooSimultaneous
        sim_pdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", {"physics": model, "control": model_ctl}, sample)

        # Verify that the PDF ends up with the right PDFs in the right categories
        self.assertEqual(sim_pdf.getPdf("physics").GetName(), "model")
        self.assertEqual(sim_pdf.getPdf("control").GetName(), "model_ctl")


if __name__ == "__main__":
    unittest.main()
