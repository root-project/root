## \file
## \ingroup tutorial_roofit_main
## \notebook -js
## Implementing the Barlow-Beeston method for taking into account the statistical
## uncertainty of a Monte-Carlo fit template.
##
## \macro_image
## \macro_code
## \macro_output
##
## Based on a demo by Wouter Verkerke
## \date June 2021
## \author Jonas Rembser, Harshal Shende, Stephan Hageboeck (C++ version)

import math

import ROOT


def create_constraints(gammas, samples, global_observables):
    constraints = ROOT.RooArgSet()
    for i, gamma in enumerate(gammas):
        nom_name = "nom_" + gamma.GetName()
        tau_name = gamma.GetName() + "_tau"
        mean_name = gamma.GetName() + "_mean"
        poisson_name = gamma.GetName() + "_constraint"

        nom_val = sum(func.dataHist().weight(i) for func in samples)

        # Like in HistFactory, the gamma of an empty bin is fixed and not
        # constrained. Otherwise, it would be a flat direction of the likelihood.
        # The other gammas get the relative Poisson uncertainty as initial error.
        if nom_val == 0.0:
            gamma.setConstant(True)
            continue
        gamma.setError(1.0 / math.sqrt(nom_val))

        nominal = ROOT.RooRealVar(nom_name, nom_name, nom_val, 0.0, ROOT.RooNumber.infinity())
        nominal.setConstant(True)
        global_observables.add(nominal)

        tau = ROOT.RooConstVar(tau_name, tau_name, nominal.getVal())

        mean = ROOT.RooProduct(mean_name, mean_name, gamma, tau)

        poisson = ROOT.RooPoisson(poisson_name, poisson_name, nominal, mean)

        # The constraint pdf takes ownership of the objects it depends on, and
        # the returned set takes ownership of the constraint pdf.
        for helper in (nominal, tau, mean):
            ROOT.SetOwnership(helper, False)
        poisson.addOwnedComponents(ROOT.RooArgSet(nominal, tau, mean))

        constraints.addOwned(poisson)
    return constraints


ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.NumIntegration)

# First, construct a likelihood model with a Gaussian signal on top of a uniform background
x = ROOT.RooRealVar("x", "x", -20, 20)
x.setBins(25)

meanG = ROOT.RooRealVar("meanG", "meanG", 1, -10, 10)
sigG = ROOT.RooRealVar("sigG", "sigG", 1.5, -10, 10)
g = ROOT.RooGaussian("g", "Gauss", x, meanG, sigG)
u = ROOT.RooUniform("u", "Uniform", x)

# Generate the data to be fitted
sigData = g.generate(x, 50)
bkgData = u.generate(x, 1000)

sumData = ROOT.RooDataSet("sumData", "Gauss + Uniform", x)
sumData.append(sigData)
sumData.append(bkgData)

# Make histogram templates for signal and background.
# Let's take a signal distribution with low statistics and a more accurate
# background distribution.
# Normally, these come from Monte Carlo simulations, but we will just generate them.
dh_sig = g.generateBinned(x, 50)
dh_bkg = u.generateBinned(x, 10000)

# ***** Case 0 - 'Rigid templates' *****

# Construct histogram shapes for signal and background
p_h_sig = ROOT.RooHistFunc("p_h_sig", "p_h_sig", x, dh_sig)
p_h_bkg = ROOT.RooHistFunc("p_h_bkg", "p_h_bkg", x, dh_bkg)

# Construct scale factors for adding the two distributions
Asig0 = ROOT.RooRealVar("Asig", "Asig", 1, 0.01, 5000)
Abkg0 = ROOT.RooRealVar("Abkg", "Abkg", 1, 0.01, 5000)

# Construct the sum model
model0 = ROOT.RooRealSumPdf("model0", "model0", [p_h_sig, p_h_bkg], [Asig0, Abkg0], True)

# ***** Case 1 - 'Barlow Beeston' *****

# Construct parameterized histogram shapes for signal and background
gammas_sig1 = ROOT.ParamHistFunc.createParamSet("p_ph_sig_gamma", x.numBins(), 0, 10)
gammas_bkg1 = ROOT.ParamHistFunc.createParamSet("p_ph_bkg_gamma", x.numBins(), 0, 10)

p_phf_sig1 = ROOT.ParamHistFunc("p_phf_sig1", "p_phf_sig1", [x], gammas_sig1)
p_phf_bkg1 = ROOT.ParamHistFunc("p_phf_bkg1", "p_phf_bkg1", [x], gammas_bkg1)

p_p_sig1 = ROOT.RooProduct("p_p_sig1", "p_p_sig1", p_h_sig, p_phf_sig1)
p_p_bkg1 = ROOT.RooProduct("p_p_bkg1", "p_p_bkg1", p_h_bkg, p_phf_bkg1)

Asig1 = ROOT.RooRealVar("Asig", "Asig", 1, 0.01, 5000)
Abkg1 = ROOT.RooRealVar("Abkg", "Abkg", 1, 0.01, 5000)

# Construct the sum of these
model1_tmp = ROOT.RooRealSumPdf("sp_ph", "sp_ph", [p_p_sig1, p_p_bkg1], [Asig1, Abkg1], True)

# Construct the subsidiary poisson measurements constraining the histogram parameters
# These ensure that the bin contents of the histograms are only allowed to vary within
# the statistical uncertainty of the Monte Carlo.
globs1 = ROOT.RooArgSet()
constraints_sig1 = create_constraints(gammas_sig1, [p_h_sig], globs1)
constraints_bkg1 = create_constraints(gammas_bkg1, [p_h_bkg], globs1)

# Construct the joint model with template PDFs and constraints
model1 = ROOT.RooProdPdf("model1", "model1", [model1_tmp, *constraints_sig1, *constraints_bkg1])

# ***** Case 2 - 'Barlow Beeston' light (one parameter per bin for all samples) *****

# Construct the histogram shapes, using the same parameters for signal and background.
# There is one parameter per bin, such as `p_ph_gamma_bin_0`. This allows bin 0 to
# fluctuate up and down. The SAME ParamHistFunc scales both the signal and the background
# histogram, so the bins fluctuate synchronously. This reduces the number of parameters.
gammas2 = ROOT.ParamHistFunc.createParamSet("p_ph_gamma", x.numBins(), 0, 10)

p_phf2 = ROOT.ParamHistFunc("p_phf2", "p_phf2", [x], gammas2)

p_p_sig2 = ROOT.RooProduct("p_p_sig2", "p_p_sig2", p_h_sig, p_phf2)
p_p_bkg2 = ROOT.RooProduct("p_p_bkg2", "p_p_bkg2", p_h_bkg, p_phf2)

Asig2 = ROOT.RooRealVar("Asig", "Asig", 1, 0.01, 5000)
Abkg2 = ROOT.RooRealVar("Abkg", "Abkg", 1, 0.01, 5000)

# As before, construct the sum of signal2 and background2
model2_tmp = ROOT.RooRealSumPdf("sp_ph", "sp_ph", [p_p_sig2, p_p_bkg2], [Asig2, Abkg2], True)

# Construct the subsidiary poisson measurements constraining the statistical fluctuations
globs2 = ROOT.RooArgSet()
constraints2 = create_constraints(gammas2, [p_h_sig, p_h_bkg], globs2)

# Construct the joint model
model2 = ROOT.RooProdPdf("model2", "model2", [model2_tmp, *constraints2])

# ************ Fit all models to data and plot *********************

result0 = model0.fitTo(sumData, PrintLevel=0, Save=True)
result1 = model1.fitTo(sumData, PrintLevel=0, Save=True, GlobalObservables=globs1)
result2 = model2.fitTo(sumData, PrintLevel=0, Save=True, GlobalObservables=globs2)

can = ROOT.TCanvas("can", "", 1500, 600)
can.Divide(3, 1)

pt = ROOT.TPaveText(-19.5, 1, -2, 25)
pt.SetFillStyle(0)
pt.SetBorderSize(0)

can.cd(1)
frame = x.frame(Title="No template uncertainties")
# Plot data to enable automatic determination of model0 normalisation:
sumData.plotOn(frame)
model0.plotOn(frame, LineColor="b", VisualizeError=result0)
# Plot data again to show it on top of model0 error bands:
sumData.plotOn(frame)
# Plot model components
model0.plotOn(frame, LineColor="b")
model0.plotOn(frame, Components=[p_h_sig], LineColor="kAzure")
model0.plotOn(frame, Components=[p_h_bkg], LineColor="r")
model0.paramOn(frame)

sigData.plotOn(frame, MarkerColor="b")
frame.Draw()

for text in [
    "No template uncertainties",
    "are taken into account.",
    "This leads to low errors",
    "for the parameters A, since",
    "the only source of errors",
    "are the data statistics.",
]:
    pt.AddText(text)
pt.DrawClone()

can.cd(2)
frame = x.frame(Title="Barlow Beeston for Sig & Bkg separately")
sumData.plotOn(frame)
model1.plotOn(frame, LineColor="b", VisualizeError=result1)
# Plot data again to show it on top of error bands:
sumData.plotOn(frame)
model1.plotOn(frame, LineColor="b")
model1.plotOn(frame, Components=[p_p_sig1], LineColor="kAzure")
model1.plotOn(frame, Components=[p_p_bkg1], LineColor="r")
model1.paramOn(frame, Parameters=[Asig1, Abkg1])

sigData.plotOn(frame, MarkerColor="b")
frame.Draw()

pt.Clear()
for text in [
    "With gamma parameters, the",
    "signal & background templates",
    "can adapt to the data.",
    "Note how the blue signal",
    "template changes its shape.",
    "This leads to higher errors",
    "of the scale parameters A.",
]:
    pt.AddText(text)
pt.DrawClone()

can.cd(3)
frame = x.frame(Title="Barlow Beeston light for (Sig+Bkg)")
sumData.plotOn(frame)
model2.plotOn(frame, LineColor="b", VisualizeError=result2)
# Plot data again to show it on top of model0 error bands:
sumData.plotOn(frame)
model2.plotOn(frame, LineColor="b")
model2.plotOn(frame, Components=[p_p_sig2], LineColor="kAzure")
model2.plotOn(frame, Components=[p_p_bkg2], LineColor="r")
model2.paramOn(frame, Parameters=[Asig2, Abkg2])

sigData.plotOn(frame, MarkerColor="b")
frame.Draw()

pt.Clear()
for text in [
    "When signal and background",
    "template share one gamma para-",
    "meter per bin, they adapt less.",
    "The errors of the A parameters",
    "also shrink slightly.",
]:
    pt.AddText(text)
pt.DrawClone()

print(f"Asig [normal ] = {Asig0.getVal()} +/- {Asig0.getError()}")
print(f"Asig [BB     ] = {Asig1.getVal()} +/- {Asig1.getError()}")
print(f"Asig [BBlight] = {Asig2.getVal()} +/- {Asig2.getError()}")

can.SaveAs("rf709_BarlowBeeston.png")
