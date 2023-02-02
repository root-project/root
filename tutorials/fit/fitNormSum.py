## \file
## \ingroup tutorial_fit
## \notebook
## Tutorial for normalized sum of two functions
## Here: a background exponential and a crystalball function
## Parameters can be set:
##  1.   with the TF1 object before adding the function (for 3) and 4))
##  2.  with the TF1NormSum object (first two are the coefficients, then the non constant parameters)
##  3. with the TF1 object after adding the function
##
## Sum can be constructed by:
##  1. by a string containing the names of the functions and/or the coefficient in front
##  2. by a string containg formulas like expo, gaus...
##  3. by the list of functions and coefficients (which are 1 by default)
##  4. by a std::vector for functions and coefficients
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Jonas Rembser, Lorenzo Moneta (C++ version)

import ROOT

nsig = 50000
nbkg = 1000000
nEvents = nsig + nbkg
nBins = 1000

signal_mean = 3.0
f_cb = ROOT.TF1("MyCrystalBall", "crystalball", -5.0, 5.0)
f_exp = ROOT.TF1("MyExponential", "expo", -5.0, 5.0)

# I.:
f_exp.SetParameters(1.0, -0.3)
f_cb.SetParameters(1, signal_mean, 0.3, 2, 1.5)

# CONSTRUCTION OF THE TF1NORMSUM OBJECT ........................................
# 1) :
fnorm_exp_cb = ROOT.TF1NormSum(f_cb, f_exp, nsig, nbkg)
# 4) :

f_sum = ROOT.TF1("fsum", fnorm_exp_cb, -5.0, 5.0, fnorm_exp_cb.GetNpar())

# III.:
parameter_values = fnorm_exp_cb.GetParameters()
f_sum.SetParameters(parameter_values.data())
# Note: in the C++ tutorial, the parameter value sync is done in one line with:
# ```C++
#     f_sum->SetParameters(fnorm_exp_cb->GetParameters().data());
# ```
# However, TF1NormSum::GetParameters() returns an std::vector by value, which
# doesn't survive long enough in Python. That's why we have to explicitly
# assign it to a variable first and can't use a temporary.

f_sum.SetParName(1, "NBackground")
f_sum.SetParName(0, "NSignal")
for i in range(2, f_sum.GetNpar()):
    f_sum.SetParName(i, fnorm_exp_cb.GetParName(i))

# GENERATE HISTOGRAM TO FIT ..............................................................
w = ROOT.TStopwatch()
w.Start()
h_sum = ROOT.TH1D("h_ExpCB", "Exponential Bkg + CrystalBall function", nBins, -5.0, 5.0)
h_sum.FillRandom("fsum", nEvents)
print("Time to generate {0} events:  ".format(nEvents))
w.Print()

# need to scale histogram with width since we are fitting a density
h_sum.Sumw2()
h_sum.Scale(1.0, "width")

# fit - use Minuit2 if available
ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2")
c1 = ROOT.TCanvas("Fit", "Fit", 800, 1000)
# do a least-square fit of the spectrum
result = h_sum.Fit("fsum", "SQ")
result.Print()
h_sum.Draw()
print("Time to fit using ROOT TF1Normsum: ")
w.Print()

# test if parameters are fine
for i, pref in enumerate([nsig, nbkg, signal_mean]):
    if not ROOT.TMath.AreEqualAbs(pref, f_sum.GetParameter(i), f_sum.GetParError(i) * 10.0):
        ROOT.Error(
            "testFitNormSum",
            "Difference found in fitted {0} - difference is {1:.2f} sigma".format(
                f_sum.GetParName(i), (f_sum.GetParameter(i) - pref) / f_sum.GetParError(i)
            ),
        )

ROOT.gStyle.SetOptStat(0)
# add parameters
t1 = ROOT.TLatex(-2.5, 300000, "NSignal = {0:g} #pm {1:g}".format(f_sum.GetParameter(0), f_sum.GetParError(0)))
t2 = ROOT.TLatex(-2.5, 270000, "Nbackgr = {0:g} #pm {1:g}".format(f_sum.GetParameter(1), f_sum.GetParError(1)))
t1.Draw()
t2.Draw()

c1.SaveAs("fitNormSum.png")
