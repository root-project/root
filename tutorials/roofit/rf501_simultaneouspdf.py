#####################################
#
# 'ORGANIZATION AND SIMULTANEOUS FITS' ROOT.RooFit tutorial macro #501
#
# Using simultaneous p.d.f.s to describe simultaneous fits to multiple
# datasets
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf501_simultaneouspdf():
    # C r e a t e   m o d e l   f o r   p h y s i c s   s a m p l e
    # -------------------------------------------------------------

    # Create observables
    x = ROOT.RooRealVar("x", "x", -8, 8)

    # Construct signal pdf
    mean = ROOT.RooRealVar("mean", "mean", 0, -8, 8)
    sigma = ROOT.RooRealVar("sigma", "sigma", 0.3, 0.1, 10)
    gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

    # Construct background pdf
    a0 = ROOT.RooRealVar("a0", "a0", -0.1, -1, 1)
    a1 = ROOT.RooRealVar("a1", "a1", 0.004, -1, 1)
    px = ROOT.RooChebychev("px", "px", x, ROOT.RooArgList(a0, a1))

    # Construct composite pdf
    f = ROOT.RooRealVar("f", "f", 0.2, 0., 1.)
    model = ROOT.RooAddPdf(
        "model", "model", ROOT.RooArgList(gx, px), ROOT.RooArgList(f))

    # C r e a t e   m o d e l   f o r   c o n t r o l   s a m p l e
    # --------------------------------------------------------------

    # Construct signal pdf.
    # NOTE that sigma is shared with the signal sample model
    mean_ctl = ROOT.RooRealVar("mean_ctl", "mean_ctl", -3, -8, 8)
    gx_ctl = ROOT.RooGaussian("gx_ctl", "gx_ctl", x, mean_ctl, sigma)

    # Construct the background pdf
    a0_ctl = ROOT.RooRealVar("a0_ctl", "a0_ctl", -0.1, -1, 1)
    a1_ctl = ROOT.RooRealVar("a1_ctl", "a1_ctl", 0.5, -0.1, 1)
    px_ctl = ROOT.RooChebychev(
        "px_ctl", "px_ctl", x, ROOT.RooArgList(a0_ctl, a1_ctl))

    # Construct the composite model
    f_ctl = ROOT.RooRealVar("f_ctl", "f_ctl", 0.5, 0., 1.)
    model_ctl = ROOT.RooAddPdf(
        "model_ctl", "model_ctl", ROOT.RooArgList(gx_ctl, px_ctl), ROOT.RooArgList(f_ctl))

    # G e n e r a t e   e v e n t s   f o r   b o t h   s a m p l e s
    # ---------------------------------------------------------------

    # Generate 1000 events in x and y from model
    data = model.generate(ROOT.RooArgSet(x), 100)
    data_ctl = model_ctl.generate(ROOT.RooArgSet(x), 2000)

    # C r e a t e   i n d e x   c a t e g o r y   a n d   j o i n   s a m p l e s
    # ---------------------------------------------------------------------------

    # Define category to distinguish physics and control samples events
    sample = ROOT.RooCategory("sample", "sample")
    sample.defineType("physics")
    sample.defineType("control")

    # Construct combined dataset in (x,sample)
    combData = ROOT.RooDataSet("combData", "combined data", ROOT.RooArgSet(x), ROOT.RooFit.Index(
        sample), ROOT.RooFit.Import("physics", data), ROOT.RooFit.Import("control", data_ctl))

    # C o n s t r u c t   a   s i m u l t a n e o u s   p d f   i n   ( x , a m p l e )
    # -----------------------------------------------------------------------------------

    # Construct a simultaneous pdf using category sample as index
    simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", sample)

    # Associate model with the physics state and model_ctl with the control
    # state
    simPdf.addPdf(model, "physics")
    simPdf.addPdf(model_ctl, "control")

    # P e r f o r m   a   s i m u l t a n e o u s   f i t
    # ---------------------------------------------------

    # Perform simultaneous fit of model to data and model_ctl to data_ctl
    simPdf.fitTo(combData)

    # P l o t   m o d e l   s l i c e s   o n   d a t a    s l i c e s
    # ----------------------------------------------------------------

    # Make a frame for the physics sample
    frame1 = x.frame(ROOT.RooFit.Bins(30), ROOT.RooFit.Title("Physics sample"))

    # Plot all data tagged as physics sample
    combData.plotOn(frame1, ROOT.RooFit.Cut("sample==sample::physics"))

    # Plot "physics" slice of simultaneous pdf.
    # NBL You _must_ project the sample index category with data using ProjWData
    # as a ROOT.RooSimultaneous makes no prediction on the shape in the index category
    # and can thus not be integrated
    simPdf.plotOn(frame1, ROOT.RooFit.Slice(sample, "physics"),
                  ROOT.RooFit.ProjWData(ROOT.RooArgSet(sample), combData))
    simPdf.plotOn(frame1, ROOT.RooFit.Slice(sample, "physics"), ROOT.RooFit.Components(
        "px"), ROOT.RooFit.ProjWData(ROOT.RooArgSet(sample), combData), ROOT.RooFit.LineStyle(ROOT.kDashed))

    # ROOT.The same plot for the control sample slice
    frame2 = x.frame(ROOT.RooFit.Bins(30), ROOT.RooFit.Title("Control sample"))
    combData.plotOn(frame2, ROOT.RooFit.Cut("sample==sample::control"))
    simPdf.plotOn(frame2, ROOT.RooFit.Slice(sample, "control"),
                  ROOT.RooFit.ProjWData(ROOT.RooArgSet(sample), combData))
    simPdf.plotOn(frame2, ROOT.RooFit.Slice(sample, "control"), ROOT.RooFit.Components(
        "px_ctl"), ROOT.RooFit.ProjWData(ROOT.RooArgSet(sample), combData), ROOT.RooFit.LineStyle(ROOT.kDashed))

    c = ROOT.TCanvas("rf501_simultaneouspdf",
                     "rf501_simultaneouspdf", 800, 400)
    c.Divide(2)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame1.GetYaxis().SetTitleOffset(1.4)
    frame1.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.4)
    frame2.Draw()

    c.SaveAs("rf501_simultaneouspdf.png")


if __name__ == "__main__":
    rf501_simultaneouspdf()
