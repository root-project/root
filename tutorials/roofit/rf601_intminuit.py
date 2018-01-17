# /
#
# 'LIKELIHOOD AND MINIMIZATION' ROOT.RooFit tutorial macro #601
#
# Interactive minimization with MINUIT
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf601_intminuit():
    # S e t u p   p d f   a n d   l i k e l i h o o d
    # -----------------------------------------------

    # Observable
    x = ROOT.RooRealVar("x", "x", -20, 20)

    # Model (intentional strong correlations)
    mean = ROOT.RooRealVar("mean", "mean of g1 and g2", 0)
    sigma_g1 = ROOT.RooRealVar("sigma_g1", "width of g1", 3)
    g1 = ROOT.RooGaussian("g1", "g1", x, mean, sigma_g1)

    sigma_g2 = ROOT.RooRealVar("sigma_g2", "width of g2", 4, 3.0, 6.0)
    g2 = ROOT.RooGaussian("g2", "g2", x, mean, sigma_g2)

    frac = ROOT.RooRealVar("frac", "frac", 0.5, 0.0, 1.0)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(g1, g2), ROOT.RooArgList(frac))

    # Generate 1000 events
    data = model.generate(ROOT.RooArgSet(x), 1000)

    # Construct unbinned likelihood of model w.r.t. data
    nll = model.createNLL(data)

    # I n t e r a c t i v e   m i n i m i z a t i o n , r r o r   a n a l y s i s
    # -------------------------------------------------------------------------------

    # Create MINUIT interface object
    m = ROOT.RooMinuit(nll)

    # Activate verbose logging of MINUIT parameter space stepping
    m.setVerbose(ROOT.kTRUE)

    # Call MIGRAD to minimize the likelihood
    m.migrad()

    # Print values of all parameters, reflect values (and error estimates)
    # that are back propagated from MINUIT
    model.getParameters(ROOT.RooArgSet(x)).Print("s")

    # Disable verbose logging
    m.setVerbose(ROOT.kFALSE)

    # Run HESSE to calculate errors from d2L/dp2
    m.hesse()

    # Print value (and error) of sigma_g2 parameter, reflects
    # value and error back propagated from MINUIT
    sigma_g2.Print()

    # Run MINOS on sigma_g2 parameter only
    m.minos(ROOT.RooArgSet(sigma_g2))

    # Print value (and error) of sigma_g2 parameter, reflects
    # value and error back propagated from MINUIT
    sigma_g2.Print()

    # S a v i n g   r e s u l t s , o n t o u r   p l o t s
    # ---------------------------------------------------------

    # Save a snapshot of the fit result. ROOT.This object contains the initial
    # fit parameters, final fit parameters, complete correlation
    # matrix, EDM, minimized FCN , last MINUIT status code and
    # the number of times the ROOT.RooFit function object has indicated evaluation
    # problems (e.g. zero probabilities during likelihood evaluation)
    r = m.save()

    # Make contour plot of mx vs sx at 1,2, sigma
    frame = m.contour(frac, sigma_g2, 1, 2, 3)
    frame.SetTitle("RooMinuit contour plot")

    # Print the fit result snapshot
    r.Print("v")

    # C h a n g e   p a r a m e t e r   v a l u e s , l o a t i n g
    # -----------------------------------------------------------------

    # At any moment you can manually change the value of a (constant)
    # parameter
    mean.setVal(0.3)

    # Rerun MIGRAD,HESSE
    m.migrad()
    m.hesse()
    frac.Print()

    # Now fix sigma_g2
    sigma_g2.setConstant(ROOT.kTRUE)

    # Rerun MIGRAD,HESSE
    m.migrad()
    m.hesse()
    frac.Print()

    c = ROOT.TCanvas("rf601_intminuit", "rf601_intminuit", 600, 600)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.Draw()

    c.SaveAs("rf601_intminuit.png")


if __name__ == "__main__":
    rf601_intminuit()
