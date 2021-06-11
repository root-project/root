## \file
## \ingroup tutorial_roofit
## \notebook
## 'ADDITION AND CONVOLUTION' RooFit tutorial macro #211
## Working a with a p.d.f. with a convolution operator in terms
## of a parameter
##
## (require ROOT to be compiled with --enable-fftw3)
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)

import ROOT

# Set up component pdfs
# ---------------------------------------

# Gaussian g(x ; mean,sigma)
x = ROOT.RooRealVar("x", "x", -10, 10)
mean = ROOT.RooRealVar("mean", "mean", -3, 3)
sigma = ROOT.RooRealVar("sigma", "sigma", 0.5, 0.1, 10)
modelx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

# Block function in mean
a = ROOT.RooRealVar("a", "a", 2, 1, 10)
model_mean = ROOT.RooGenericPdf("model_mean", "abs(mean)<a", ROOT.RooArgList(mean, a))

# Convolution in mean model = g(x,mean,sigma) (x) block(mean)
x.setBins(1000, "cache")
mean.setBins(50, "cache")
model = ROOT.RooFFTConvPdf("model", "model", mean, modelx, model_mean)

# Configure convolution to construct a 2-D cache in (x,mean)
# rather than a 1-d cache in mean that needs to be recalculated
# for each value of x
model.setCacheObservables(ROOT.RooArgSet(x))
model.setBufferFraction(1.0)

# Integrate model over projModel = Int model dmean
projModel = model.createProjection(ROOT.RooArgSet(mean))

# Generate 1000 toy events
d = projModel.generateBinned(ROOT.RooArgSet(x), 1000)

# Fit p.d.f. to toy data
projModel.fitTo(d, Verbose=True)

# Plot data and fitted p.d.f.
frame = x.frame(ROOT.RooFit.Bins(25))
d.plotOn(frame)
projModel.plotOn(frame)

# Make 2d histogram of model(x;mean)
hh = model.createHistogram(
    "hh",
    x,
    ROOT.RooFit.Binning(50),
    ROOT.RooFit.YVar(mean, ROOT.RooFit.Binning(50)),
    ROOT.RooFit.ConditionalObservables(ROOT.RooArgSet(mean)),
)
hh.SetTitle("histogram of model(x|mean)")
hh.SetLineColor(ROOT.kBlue)

# Draw frame on canvas
c = ROOT.TCanvas("rf211_paramconv", "rf211_paramconv", 800, 400)
c.Divide(2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame.GetYaxis().SetTitleOffset(1.4)
frame.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.20)
hh.GetZaxis().SetTitleOffset(2.5)
hh.Draw("surf")

c.SaveAs("rf211_paramconv.png")
