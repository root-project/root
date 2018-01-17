#####################################
#
# 'MULTIDIMENSIONAL MODELS' ROOT.RooFit tutorial macro #309
#
# Projecting p.d.f and data slices in discrete observables
#
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf310_sliceplot():

    # C r e a t e   B   d e c a y   p d f   w it h   m i x i n g
    # ----------------------------------------------------------

    # Decay time observables
    dt = ROOT.RooRealVar("dt", "dt", -20, 20)

    # Discrete observables mixState (B0tag==B0reco?) and tagFlav
    # (B0tag==B0(bar)?)
    mixState = ROOT.RooCategory("mixState", "B0/B0bar mixing state")
    tagFlav = ROOT.RooCategory("tagFlav", "Flavour of the tagged B0")

    # Define state labels of discrete observables
    mixState.defineType("mixed", -1)
    mixState.defineType("unmixed", 1)
    tagFlav.defineType("B0", 1)
    tagFlav.defineType("B0bar", -1)

    # Model parameters
    dm = ROOT.RooRealVar("dm", "delta m(B)", 0.472, 0., 1.0)
    tau = ROOT.RooRealVar("tau", "B0 decay time", 1.547, 1.0, 2.0)
    w = ROOT.RooRealVar("w", "Flavor Mistag rate", 0.03, 0.0, 1.0)
    dw = ROOT.RooRealVar(
        "dw", "Flavor Mistag rate difference between B0 and B0bar", 0.01)

    # Build a gaussian resolution model
    bias1 = ROOT.RooRealVar("bias1", "bias1", 0)
    sigma1 = ROOT.RooRealVar("sigma1", "sigma1", 0.01)
    gm1 = ROOT.RooGaussModel("gm1", "gauss model 1", dt, bias1, sigma1)

    # Construct a decay pdf, with single gaussian resolution model
    bmix_gm1 = ROOT.RooBMixDecay("bmix", "decay", dt, mixState,
                                 tagFlav, tau, dm, w, dw, gm1, ROOT.RooBMixDecay.DoubleSided)

    # Generate BMixing data with above set of event errors
    data = bmix_gm1.generate(ROOT.RooArgSet(dt, tagFlav, mixState), 20000)

    # P l o t   f u l l   d e c a y   d i s t r i b u t i o n
    # ----------------------------------------------------------

    # Create frame, data and pdf projection (integrated over tagFlav and
    # mixState)
    frame = dt.frame(ROOT.RooFit.Title("Inclusive decay distribution"))
    data.plotOn(frame)
    bmix_gm1.plotOn(frame)

    # P l o t   d e c a y   d i s t r .   f o r   m i x e d   a n d   u n m i x e d   s l i c e   o f   m i x S t a t e
    # ------------------------------------------------------------------------------------------------------------------

    # Create frame, data (mixed only)
    frame2 = dt.frame(ROOT.RooFit.Title("Decay distribution of mixed events"))
    data.plotOn(frame2, ROOT.RooFit.Cut("mixState==mixState.mixed"))

    # Position slice in mixState at "mixed" and plot slice of pdf in mixstate
    # over data (integrated over tagFlav)
    bmix_gm1.plotOn(frame2, ROOT.RooFit.Slice(mixState, "mixed"))

    # Create frame, data (unmixed only)
    frame3 = dt.frame(ROOT.RooFit.Title(
        "Decay distribution of unmixed events"))
    data.plotOn(frame3, ROOT.RooFit.Cut("mixState==mixState.unmixed"))

    # Position slice in mixState at "unmixed" and plot slice of pdf in
    # mixstate over data (integrated over tagFlav)
    bmix_gm1.plotOn(frame3, ROOT.RooFit.Slice(mixState, "unmixed"))

    c = ROOT.TCanvas("rf310_sliceplot", "rf310_sliceplot", 1200, 400)
    c.Divide(3)
    c.cd(1)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.4)
    ROOT.gPad.SetLogy()
    frame.Draw()
    c.cd(2)
    ROOT.gPad.SetLeftMargin(0.15)
    frame2.GetYaxis().SetTitleOffset(1.4)
    ROOT.gPad.SetLogy()
    frame2.Draw()
    c.cd(3)
    ROOT.gPad.SetLeftMargin(0.15)
    frame3.GetYaxis().SetTitleOffset(1.4)
    ROOT.gPad.SetLogy()
    frame3.Draw()

    c.SaveAs("rf310_sliceplot.png")


if __name__ == "__main__":
    rf310_sliceplot()
