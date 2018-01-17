# /
#
# 'ADDITION AND CONVOLUTION' ROOT.RooFit tutorial macro #208
#
# One-dimensional numeric convolution
# (require ROOT to be compiled with --enable-fftw3)
#
# pdf = landau(t) (x) gauss(t)
#
#
# 07/2008 - Wouter Verkerke
#
# /


import ROOT


def rf208_convolution():
    # S e t u p   c o m p o n e n t   p d f s
    # ---------------------------------------

    # Construct observable
    t = ROOT.RooRealVar("t", "t", -10, 30)

    # Construct landau(t,ml,sl)
    ml = ROOT.RooRealVar("ml", "mean landau", 5., -20, 20)
    sl = ROOT.RooRealVar("sl", "sigma landau", 1, 0.1, 10)
    landau = ROOT.RooLandau("lx", "lx", t, ml, sl)

    # Construct gauss(t,mg,sg)
    mg = ROOT.RooRealVar("mg", "mg", 0)
    sg = ROOT.RooRealVar("sg", "sg", 2, 0.1, 10)
    gauss = ROOT.RooGaussian("gauss", "gauss", t, mg, sg)

    # C o n s t r u c t   c o n v o l u t i o n   p d f
    # ---------------------------------------

    # Set #bins to be used for FFT sampling to 10000
    t.setBins(10000, "cache")

    # Construct landau (x) gauss
    lxg = ROOT.RooFFTConvPdf("lxg", "landau (X) gauss", t, landau, gauss)

    # S a m p l e , i t   a n d   p l o t   c o n v o l u t e d   p d f
    # ----------------------------------------------------------------------

    # Sample 1000 events in x from gxlx
    data = lxg.generate(ROOT.RooArgSet(t), 10000)

    # Fit gxlx to data
    lxg.fitTo(data)

    # Plot data, pdf, landau (X) gauss pdf
    frame = t.frame(ROOT.RooFit.Title("landau (x) gauss convolution"))
    data.plotOn(frame)
    lxg.plotOn(frame)
    landau.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed))

    # Draw frame on canvas
    c = ROOT.TCanvas("rf208_convolution", "rf208_convolution", 600, 600)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.Draw()

    c.SaveAs("rf208_convolution.png")


if __name__ == "__main__":
    rf208_convolution()
