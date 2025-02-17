import unittest

import ROOT


class XRooFitTests(unittest.TestCase):

    def test_oneChannelLimit(self):
        """
        Tests creating a workspace (from histograms) containing pdf and dataset
        for a single-channel, 3-bin model, with sig and bkg samples
        and computing the CLs upper limit on the signal strength POI
        """

        # create some histograms to represent the samples, syst variations, and obsData
        from array import array

        bkg = ROOT.TH1D("bkg", "Background", 3, 0, 3)
        bkg.SetContent(array("d", [0, 15, 14, 13, 0]))  # first and last are under/overflow
        sig = ROOT.TH1D("sig", "Signal", 3, 0, 3)
        sig.SetContent(array("d", [0, 4, 5, 6, 0]))
        bkg_vary1 = bkg.Clone("alphaSyst=1")
        bkg_vary1.SetContent(array("d", [0, 16, 13, 12, 0]))
        obsData = ROOT.TH1D("obsData", "Data", 3, 0, 3)
        obsData.SetContent(array("d", [0, 17, 15, 13, 0]))
        bkg.SetFillColor(ROOT.kRed)  # can e.g. add style settings to histograms, they will propagate into the workspace

        import ROOT.Experimental.XRooFit as XRF

        # create workspace
        w = XRF.xRooNode("RooWorkspace", name="combined", title="combined")
        # create a pdf and add a "SR" channel to it
        pdf = w["pdfs"].Add("simPdf")
        sr = pdf.Add("SR")
        # add our samples to the channel
        sr_bkg = sr.Add(bkg)
        sr_sig = sr.Add(sig)
        # add the variation on the bkg sample
        sr_bkg.Vary(bkg_vary1)
        # constrain the nuisance parameter that was created
        pdf.pars()["alphaSyst"].Constrain("normal")  # normal gaussian constraint
        # create a signal strength POI and scale the sig term by it
        w.poi().Add("mu[1]")
        sr_sig.Multiply("mu")
        # add the obsData to the channel
        sr.datasets().Add(obsData)

        # example of accessing expected yields in bins with propagated errors                                                                                                 
        w.poi()["mu"].setVal(0)
        self.assertAlmostEqual(w["pdfs/simPdf/SR"].GetContent(), bkg.Integral())
        self.assertAlmostEqual(w["pdfs/simPdf/SR"].GetError(), abs(bkg_vary1.Integral()-bkg.Integral()) )
        # accessing a single sample expected yield                                                                                                                            
        w.poi()["mu"].setVal(0.5)
        self.assertAlmostEqual(w["pdfs/simPdf/SR/sig"].GetContent(), sig.Integral()*w.poi()["mu"].getVal())
        self.assertAlmostEqual(w["pdfs/simPdf/SR/sig"].GetError(), 0.) # no uncert was added to the signal

        # could save the workspace as this point like this:
        # w.SaveAs("ws_test_oneChannelLimit.root")

        # run a limit by creating a hypoSpace for the POI
        hs = w.nll("obsData").hypoSpace("mu")
        # when cls limit scan happens, fits are run. If we have an open writable TFile, all results will cache there
        f = ROOT.TFile("fitCache.root", "RECREATE")
        hs.scan("cls", nPoints=0, low=0, high=10)  # nPoints=0 means will do an auto-scan for the limit
        f.Close()
        limits = dict(hs.limits())  # accesses the limits in the form of a dict

        for k, v in limits.items():
            print(k, "sigma expected limit =" if k != "obs" else "observed limit =", v.value(), "+/-", v.error())
            # do a basic check that all values and errors are valid
            # and nan values are indicative of problems
            assert not ROOT.TMath.IsNaN(v.value())
            assert not ROOT.TMath.IsNaN(v.error())

        # example code for accessing the fit results of the fits that were run as part of the computation
        def printInfo(hp, prefix=""):
            print(prefix, "null mu=", hp.fNullVal(), "alt mu=", hp.fAltVal())
            fitResults = {
                "ufit": hp.ufit(readOnly=True),  # readOnly ensures wont try to do the fit if it wasn't needed
                "cfit_null": hp.cfit_null(readOnly=True),  # conditional fit with null mu value
                "cfit_alt": hp.cfit_alt(readOnly=True),  # conditional fit with alt mu value (usually mu=0)
                "cfit_lbound": hp.cfit_lbound(readOnly=True),  # sometimes necessary for lower-bound test statistics
                "gfit": hp.gfit(),  # the fit result from which the data was generated, if the data for this point is generated
            }
            for k, v in fitResults.items():
                print(prefix, "  ", k, "status =", v.status() if v else "N/A")

        for hp in hs:  # loop over the hypoPoints in the hypoSpace
            printInfo(hp)
            if hp.asimov():
                printInfo(hp.asimov(), "  asimov")


if __name__ == "__main__":
    unittest.main()
