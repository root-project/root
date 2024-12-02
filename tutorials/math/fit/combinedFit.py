## \file
## \ingroup tutorial_fit
## \notebook
## Combined (simultaneous) fit of two histogram with separate functions
## and some common parameters
##
## \macro_image
## \macro_output
## \macro_code
##
## \author Jonas Rembser, Lorenzo Moneta (C++ version)


import ROOT
import numpy as np


# definition of shared parameter background function
iparB = np.array([0, 2], dtype=np.int32)  # exp amplitude in B histo and exp common parameter

# signal + background function
iparSB = np.array(
    [
        1,  # exp amplitude in S+B histo
        2,  # exp common parameter
        3,  # Gaussian amplitude
        4,  # Gaussian mean
        5,  # Gaussian sigma
    ],
    dtype=np.int32,
)

# Create the GlobalCHi2 structure

class GlobalChi2(object):
    def __init__(self, f1, f2):
        self._f1 = f1
        self._f2 = f2

    def __call__(self, par):
        # parameter vector is first background (in common 1 and 2) and then is
        # signal (only in 2)

        # the zero-copy way to get a numpy array from a double *
        par_arr = np.frombuffer(par, dtype=np.float64, count=6)

        p1 = par_arr[iparB]
        p2 = par_arr[iparSB]

        return self._f1(p1) + self._f2(p2)


hB = ROOT.TH1D("hB", "histo B", 100, 0, 100)
hSB = ROOT.TH1D("hSB", "histo S+B", 100, 0, 100)

fB = ROOT.TF1("fB", "expo", 0, 100)
fB.SetParameters(1, -0.05)
hB.FillRandom("fB")

fS = ROOT.TF1("fS", "gaus", 0, 100)
fS.SetParameters(1, 30, 5)

hSB.FillRandom("fB", 2000)
hSB.FillRandom("fS", 1000)

# perform now global fit

fSB = ROOT.TF1("fSB", "expo + gaus(2)", 0, 100)

wfB = ROOT.Math.WrappedMultiTF1(fB, 1)
wfSB = ROOT.Math.WrappedMultiTF1(fSB, 1)

opt = ROOT.Fit.DataOptions()
rangeB = ROOT.Fit.DataRange()
# set the data range
rangeB.SetRange(10, 90)
dataB = ROOT.Fit.BinData(opt, rangeB)
ROOT.Fit.FillData(dataB, hB)

rangeSB = ROOT.Fit.DataRange()
rangeSB.SetRange(10, 50)
dataSB = ROOT.Fit.BinData(opt, rangeSB)
ROOT.Fit.FillData(dataSB, hSB)

chi2_B = ROOT.Fit.Chi2Function(dataB, wfB)
chi2_SB = ROOT.Fit.Chi2Function(dataSB, wfSB)

globalChi2 = GlobalChi2(chi2_B, chi2_SB)

fitter = ROOT.Fit.Fitter()

Npar = 6
par0 = np.array([5, 5, -0.1, 100, 30, 10])

# create before the parameter settings in order to fix or set range on them
fitter.Config().SetParamsSettings(6, par0)
# fix 5-th parameter
fitter.Config().ParSettings(4).Fix()
# set limits on the third and 4-th parameter
fitter.Config().ParSettings(2).SetLimits(-10, -1.0e-4)
fitter.Config().ParSettings(3).SetLimits(0, 10000)
fitter.Config().ParSettings(3).SetStepSize(5)

fitter.Config().MinimizerOptions().SetPrintLevel(0)
fitter.Config().SetMinimizer("Minuit2", "Migrad")

# we can't pass the Python object globalChi2 directly to FitFCN.
# It needs to be wrapped in a ROOT::Math::Functor.
globalChi2Functor = ROOT.Math.Functor(globalChi2, 6)

# fit FCN function
# (specify optionally data size and flag to indicate that is a chi2 fit)
fitter.FitFCN(globalChi2Functor, 0, dataB.Size() + dataSB.Size(), True)
result = fitter.Result()
result.Print(ROOT.std.cout)

c1 = ROOT.TCanvas("Simfit", "Simultaneous fit of two histograms", 10, 10, 700, 700)
c1.Divide(1, 2)
c1.cd(1)
ROOT.gStyle.SetOptFit(1111)

fB.SetFitResult(result, iparB)
fB.SetRange(rangeB().first, rangeB().second)
fB.SetLineColor(ROOT.kBlue)
hB.GetListOfFunctions().Add(fB)
hB.Draw()

c1.cd(2)
fSB.SetFitResult(result, iparSB)
fSB.SetRange(rangeSB().first, rangeSB().second)
fSB.SetLineColor(ROOT.kRed)
hSB.GetListOfFunctions().Add(fSB)
hSB.Draw()

c1.SaveAs("combinedFit.png")
