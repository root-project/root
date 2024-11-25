## \file
## \ingroup tutorial_roofit_main
## \notebook -js
## A tutorial that explains you how to solve problems with binning effects and
## numerical stability in binned fits.
##
## ### Introduction
##
## In this tutorial, you will learn three new things:
##
##  1. How to reduce the bias in binned fits by changing the definition of the
##     normalization integral
##
##  2. How to completely get rid of binning effects by integrating the pdf over each bin
##
##  3. How to improve the numeric stability of fits with a greatly different
##     number of events per bin, using a constant per-bin counterterm
##
## \macro_code
## \macro_output
##
## \date January 2023
## \author Jonas Rembser

import ROOT


def generateBinnedAsimov(pdf, x, n_events):
    """
    Generate binned Asimov dataset for a continuous pdf.
    One should in principle be able to use
    pdf.generateBinned(x, n_events, RooFit::ExpectedData()).
    Unfortunately it has a problem: it also has the bin bias that this tutorial
    demonstrates, to if we would use it, the biases would cancel out.
    """
    data_h = ROOT.RooDataHist("dataH", "dataH", {x})
    x_binning = x.getBinning()

    for i_bin in range(x.numBins()):
        x.setRange("bin", x_binning.binLow(i_bin), x_binning.binHigh(i_bin))
        integ = pdf.createIntegral(x, NormSet=x, Range="bin")
        ROOT.SetOwnership(integ, True)
        integ.getVal()
        data_h.set(i_bin, n_events * integ.getVal(), -1)

    return data_h


def enableBinIntegrator(func, num_bins):
    """
    Force numeric integration and do this numeric integration with the
    RooBinIntegrator, which sums the function values at the bin centers.
    """
    custom_config = ROOT.RooNumIntConfig(func.getIntegratorConfig())
    custom_config.method1D().setLabel("RooBinIntegrator")
    custom_config.getConfigSection("RooBinIntegrator").setRealValue("numBins", num_bins)
    func.setIntegratorConfig(custom_config)
    func.forceNumInt(True)


def disableBinIntegrator(func):
    """
    Reset the integrator config to disable the RooBinIntegrator.
    """
    func.setIntegratorConfig()
    func.forceNumInt(False)


# Silence info output for this tutorial
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Minimization)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Fitting)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Generation)

# Exponential example
# -------------------

# Set up the observable
x = ROOT.RooRealVar("x", "x", 0.1, 5.1)
x.setBins(10)
# fewer bins so we have larger binning effects for this demo

# Let's first look at the example of an exponential function
c = ROOT.RooRealVar("c", "c", -1.8, -5, 5)
expo = ROOT.RooExponential("expo", "expo", x, c)

# Generate an Asimov dataset such that the only difference between the fit
# result and the true parameters comes from binning effects.
expo_data = generateBinnedAsimov(expo, x, 10000)

# If you do the fit the usual was in RooFit, you will get a bias in the
# result. This is because the continuous, normalized pdf is evaluated only
# at the bin centers.
fit1 = expo.fitTo(expo_data, Save=True, PrintLevel=-1, SumW2Error=False)
fit1.Print()

# In the case of an exponential function, the bias that you get by
# evaluating the pdf only at the bin centers is a constant scale factor in
# each bin. Here, we can do a trick to get rid of the bias: we also
# evaluate the normalization integral for the pdf the same way, i.e.,
# summing the values of the unnormalized pdf at the bin centers. Like this
# the bias cancels out. You can achieve this by customizing the way how the
# pdf is integrated (see also the rf901_numintconfig tutorial).
enableBinIntegrator(expo, x.numBins())
fit2 = expo.fitTo(expo_data, Save=True, PrintLevel=-1, SumW2Error=False)
fit2.Print()
disableBinIntegrator(expo)

# Power law example
# -----------------

# Let's not look at another example: a power law \f[x^a\f].
a = ROOT.RooRealVar("a", "a", -0.3, -5.0, 5.0)
powerlaw = ROOT.RooPowerSum("powerlaw", "powerlaw", x, ROOT.RooFit.RooConst(1.0), a)
powerlaw_data = generateBinnedAsimov(powerlaw, x, 10000)

# Again, if you do a vanilla fit, you'll get a bias
fit3 = powerlaw.fitTo(powerlaw_data, Save=True, PrintLevel=-1, SumW2Error=False)
fit3.Print()

# This time, the bias is not the same factor in each bin! This means our
# trick by sampling the integral in the same way doesn't cancel out the
# bias completely. The average bias is canceled, but there are per-bin
# biases that remain. Still, this method has some value: it is cheaper than
# rigurously correcting the bias by integrating the pdf in each bin. So if
# you know your per-bin bias variations are small or performance is an
# issue, this approach can be sufficient.
enableBinIntegrator(powerlaw, x.numBins())
fit4 = powerlaw.fitTo(powerlaw_data, Save=True, PrintLevel=-1, SumW2Error=False)
fit4.Print()
disableBinIntegrator(powerlaw)

# To get rid of the binning effects in the general case, one can use the
# IntegrateBins() command argument. Now, the pdf is not evaluated at the
# bin centers, but numerically integrated over each bin and divided by the
# bin width. The parameter for IntegrateBins() is the required precision
# for the numeric integrals. This is computationally expensive, but the
# bias is now not a problem anymore.
fit5 = powerlaw.fitTo(powerlaw_data, IntegrateBins=1e-3, Save=True, PrintLevel=-1, SumW2Error=False)
fit5.Print()

# Improving numerical stability
# -----------------------------

# There is one more problem with binned fits that is related to the binning
# effects because often, a binned fit is affected by both problems.
#
# The issue is numerical stability for fits with a greatly different number
# of events in each bin. For each bin, you have a term \f[n\log(p)\f] in
# the NLL, where \f[n\f] is the number of observations in the bin, and
# \f[p\f] the predicted probability to have an event in that bin. The
# difference in the logarithms for each bin is small, but the difference in
# \f[n\f] can be orders of magnitudes! Therefore, when summing these terms,
# lots of numerical precision is lost for the bins with less events.

# We can study this with the example of an exponential plus a Gaussian. The
# Gaussian is only a faint signal in the tail of the exponential where
# there are not so many events. And we can't afford any precision loss for
# these bins, otherwise we can't fit the Gaussian.

x.setBins(100)  # It's not about binning effects anymore, so reset the number of bins.

mu = ROOT.RooRealVar("mu", "mu", 3.0, 0.1, 5.1)
sigma = ROOT.RooRealVar("sigma", "sigma", 0.5, 0.01, 5.0)
gauss = ROOT.RooGaussian("gauss", "gauss", x, mu, sigma)

nsig = ROOT.RooRealVar("nsig", "nsig", 10000, 0, 1e9)
nbkg = ROOT.RooRealVar("nbkg", "nbkg", 10000000, 0, 1e9)
frac = ROOT.RooRealVar("frac", "frac", nsig.getVal() / (nsig.getVal() + nbkg.getVal()), 0.0, 1.0)

model = ROOT.RooAddPdf("model", "model", [gauss, expo], [nsig, nbkg])

model_data = model.generateBinned(x)

# Set the starting values for the Gaussian parameters away from the true
# value such that the fit is not trivial.
mu.setVal(2.0)
sigma.setVal(1.0)

fit6 = model.fitTo(model_data, Save=True, PrintLevel=-1, SumW2Error=False)
fit6.Print()

# You should see in the previous fit result that the fit did not converge:
# the `MINIMIZE` return code should be -1 (a successful fit has status code zero).

# To improve the situation, we can apply a numeric trick: if we subtract in
# each bin a constant counterterm \f[n\log(n/N)\f], we get terms for each
# bin that are closer to each other in order of magnitude as long as the
# initial model is not extremely off. Proving this mathematically is left
# as an exercise to the reader.

# This counterterms can be enabled by passing the Offset("bin") option to
# RooAbsPdf::fitTo() or RooAbsPdf::createNLL().

fit7 = model.fitTo(model_data, Offset="bin", Save=True, PrintLevel=-1, SumW2Error=False)
fit7.Print()

# You should now see in the last fit result that the fit has converged.
