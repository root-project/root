## \file
## \ingroup tutorial_roofit
## \notebook
## Convolution in cyclical angular observables theta, and
## construction of p.d.f in terms of trasnformed angular
## coordinates, e.g. cos(theta), the convolution
## is performed in theta rather than cos(theta)
##
## (require ROOT to be compiled with --enable-fftw3)
##
## pdf(theta)    = ROOT.T(theta)          (x) gauss(theta)
## pdf(cosTheta) = ROOT.T(acos(cosTheta)) (x) gauss(acos(cosTheta))
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange
## \author Wouter Verkerke (C version)

import ROOT

# Set up component pdfs
# ---------------------------------------

# Define angle psi
psi = ROOT.RooRealVar("psi", "psi", 0, 3.14159268)

# Define physics p.d.f T(psi)
Tpsi = ROOT.RooGenericPdf("Tpsi", "1+sin(2*@0)", [psi])

# Define resolution R(psi)
gbias = ROOT.RooRealVar("gbias", "gbias", 0.2, 0.0, 1)
greso = ROOT.RooRealVar("greso", "greso", 0.3, 0.1, 1.0)
Rpsi = ROOT.RooGaussian("Rpsi", "Rpsi", psi, gbias, greso)

# Define cos(psi) and function psif that calculates psi from cos(psi)
cpsi = ROOT.RooRealVar("cpsi", "cos(psi)", -1, 1)
psif = ROOT.RooFormulaVar("psif", "acos(cpsi)", [cpsi])

# Define physics p.d.f. also as function of cos(psi): T(psif(cpsi)) = T(cpsi)
Tcpsi = ROOT.RooGenericPdf("T", "1+sin(2*@0)", [psif])

# Construct convolution pdf in psi
# --------------------------------------------------------------

# Define convoluted p.d.f. as function of psi: M=[T(x)R](psi) = M(psi)
Mpsi = ROOT.RooFFTConvPdf("Mf", "Mf", psi, Tpsi, Rpsi)

# Set the buffer fraction to zero to obtain a ROOT.True cyclical
# convolution
Mpsi.setBufferFraction(0)

# Sample, fit and plot convoluted pdf (psi)
# --------------------------------------------------------------------------------

# Generate some events in observable psi
data_psi = Mpsi.generate({psi}, 10000)

# Fit convoluted model as function of angle psi
Mpsi.fitTo(data_psi)

# Plot cos(psi) frame with Mf(cpsi)
frame1 = psi.frame(Title="Cyclical convolution in angle psi")
data_psi.plotOn(frame1)
Mpsi.plotOn(frame1)

# Overlay comparison to unsmeared physics p.d.f ROOT.T(psi)
Tpsi.plotOn(frame1, LineColor="r")

# Construct convolution pdf in cos(psi)
# --------------------------------------------------------------------------

# Define convoluted p.d.f. as function of cos(psi): M=[T(x)R](psif cpsi)) = M(cpsi:
#
# Need to give both observable psi here (for definition of convolution)
# and function psif here (for definition of observables, in cpsi)
Mcpsi = ROOT.RooFFTConvPdf("Mf", "Mf", psif, psi, Tpsi, Rpsi)

# Set the buffer fraction to zero to obtain a ROOT.True cyclical
# convolution
Mcpsi.setBufferFraction(0)

# Sample, fit and plot convoluted pdf (cospsi)
# --------------------------------------------------------------------------------

# Generate some events
data_cpsi = Mcpsi.generate({cpsi}, 10000)

# set psi constant to exclude to be a parameter of the fit
psi.setConstant(True)

# Fit convoluted model as function of cos(psi)
Mcpsi.fitTo(data_cpsi)

# Plot cos(psi) frame with Mf(cpsi)
frame2 = cpsi.frame(Title="Same convolution in psi, in cos(psi)")
data_cpsi.plotOn(frame2)
Mcpsi.plotOn(frame2)

# Overlay comparison to unsmeared physics p.d.f ROOT.Tf(cpsi)
Tcpsi.plotOn(frame2, LineColor="r")

# Draw frame on canvas
c = ROOT.TCanvas("rf210_angularconv", "rf210_angularconv", 800, 400)
c.Divide(2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.4)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.4)
frame2.Draw()

c.SaveAs("rf210_angularconv.png")
