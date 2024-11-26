## \file
## \ingroup tutorial_fit
## \notebook
## Tutorial for convolution of two functions
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Jonas Rembser, Aurelie Flandi (C++ version)

import ROOT

# Construction of histogram to fit.
h_ExpGauss = ROOT.TH1F("h_ExpGauss", "Exponential convoluted by Gaussian", 100, 0.0, 5.0)
for i in range(1000000):
    # Gives a alpha of -0.3 in the exp.
    x = ROOT.gRandom.Exp(1.0 / 0.3)
    x += ROOT.gRandom.Gaus(0.0, 3.0)
    # Probability density function of the addition of two variables is the
    # convolution of two density functions.
    h_ExpGauss.Fill(x)

f_conv = ROOT.TF1Convolution("expo", "gaus", -1, 6, True)
f_conv.SetRange(-1.0, 6.0)
f_conv.SetNofPointsFFT(1000)
f = ROOT.TF1("f", f_conv, 0.0, 5.0, f_conv.GetNpar())
f.SetParameters(1.0, -0.3, 0.0, 1.0)

c1 = ROOT.TCanvas("c1", "c1", 800, 1000)

# Fit and draw result of the fit
h_ExpGauss.Fit("f")

c1.SaveAs("fitConvolution.png")
