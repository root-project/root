#####################################
#
# 'BASIC FUNCTIONALITY' ROOT.RooFit tutorial macro #105
#
#  Demonstration of binding ROOT Math functions as ROOT.RooFit functions
#  and pdfs
#
# 07/2008 - Wouter Verkerke
#
# /

import ROOT


def rf105_funcbinding():

    # B i n d   ROOT.T M a t h : : E r f   C   f u n c t i o n
    # ---------------------------------------------------

    # Bind one-dimensional ROOT.TMath.Erf function as ROOT.RooAbsReal function
    x = ROOT.RooRealVar("x", "x", -3, 3)
    erf = ROOT.RooFit.bindFunction("erf", ROOT.TMath.Erf, x)

    # Print erf definition
    erf.Print()

    # Plot erf on frame
    frame1 = x.frame(ROOT.RooFit.Title(
        "TMath.Erf bound as ROOT.RooFit function"))
    erf.plotOn(frame1)

    # B i n d   R O O ROOT.T : : M a t h : : b e t a _ p d f   C   f u n c t i o n
    # -----------------------------------------------------------------------

    # Bind pdf ROOT.Math.Beta with three variables as ROOT.RooAbsPdf function
    x2 = ROOT.RooRealVar("x2", "x2", 0, 0.999)
    a = ROOT.RooRealVar("a", "a", 5, 0, 10)
    b = ROOT.RooRealVar("b", "b", 2, 0, 10)
    beta = ROOT.RooFit.bindPdf("beta", ROOT.Math.beta_pdf, x2, a, b)

    # Perf beta definition
    beta.Print()

    # Generate some events and fit
    data = beta.generate(ROOT.RooArgSet(x2), 10000)
    beta.fitTo(*data)

    # Plot data and pdf on frame
    frame2 = x2.frame(ROOT.RooFit.Title(
        "ROOT.Math.Beta bound as ROOT.RooFit pdf"))
    data.plotOn(frame2)
    beta.plotOn(frame2)

    # B i n d   R O O ROOT.T   ROOT.T F 1   a s   R o o F i t   f u n c t i o n
    # ---------------------------------------------------------------

    # Create a ROOT ROOT.TF1 function
    fa1 = ROOT.TF1("fa1", "sin(x)/x", 0, 10)

    # Create an observable
    x3 = ROOT.RooRealVar("x3", "x3", 0.01, 20)

    # Create binding of ROOT.TF1 object to above observable
    rfa1 = ROOT.RooFit.bindFunction(fa1, x3)

    # Print rfa1 definition
    rfa1.Print()

    # Make plot frame in observable, ROOT.TF1 binding function
    frame3 = x3.frame(ROOT.RooFit.Title("TF1 bound as ROOT.RooFit function"))
    rfa1.plotOn(frame3)

    c = ROOT.TCanvas("rf105_funcbinding", "rf105_funcbinding", 1200, 400)
    c.Divide(3)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.6)
    frame1.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.6)
    frame2.Draw()
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.15)
    frame3.GetYaxis().SetTitleOffset(1.6)
    frame3.Draw()

    c.SaveAs("rf105_funcbinding.png")

if __name__ == "__main__":
    rf105_funcbinding()
