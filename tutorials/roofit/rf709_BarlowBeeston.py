## \file
## \ingroup tutorial_roofit
## \notebook -js
## Implementing the Barlow-Beeston method for taking into account the statistical
## uncertainty of a Monte-Carlo fit template.
##
## \macro_image
## \macro_output
## \macro_code
##
## Based on a demo by Wouter Verkerke
## \date June 2021
## \author Harshal Shende, Stephan Hageboeck (C++ version)


import ROOT

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


#  Case 0 - 'Rigid templates'

# Construct histogram shapes for signal and background
p_h_sig = ROOT.RooHistFunc("p_h_sig", "p_h_sig", x, dh_sig)
p_h_bkg = ROOT.RooHistFunc("p_h_bkg", "p_h_bkg", x, dh_bkg)

# Construct scale factors for adding the two distributions
Asig0 = ROOT.RooRealVar("Asig", "Asig", 1, 0.01, 5000)
Abkg0 = ROOT.RooRealVar("Abkg", "Abkg", 1, 0.01, 5000)

# Construct the sum model
model0 = ROOT.RooRealSumPdf("model0", "model0", ROOT.RooArgList(p_h_sig, p_h_bkg), ROOT.RooArgList(Asig0, Abkg0), True)


#  Case 1 - 'Barlow Beeston'

# Construct parameterized histogram shapes for signal and background
p_ph_sig1 = ROOT.RooParamHistFunc("p_ph_sig", "p_ph_sig", dh_sig)
p_ph_bkg1 = ROOT.RooParamHistFunc("p_ph_bkg", "p_ph_bkg", dh_bkg)

Asig1 = ROOT.RooRealVar("Asig", "Asig", 1, 0.01, 5000)
Abkg1 = ROOT.RooRealVar("Abkg", "Abkg", 1, 0.01, 5000)

# Construct the sum of these
model_tmp = ROOT.RooRealSumPdf(
    "sp_ph", "sp_ph", ROOT.RooArgList(p_ph_sig1, p_ph_bkg1), ROOT.RooArgList(Asig1, Abkg1), True
)

# Construct the subsidiary poisson measurements constraining the histogram parameters
# These ensure that the bin contents of the histograms are only allowed to vary within
# the statistical uncertainty of the Monte Carlo.
hc_sig = ROOT.RooHistConstraint("hc_sig", "hc_sig", p_ph_sig1)
hc_bkg = ROOT.RooHistConstraint("hc_bkg", "hc_bkg", p_ph_bkg1)

# Construct the joint model with template PDFs and constraints
model1 = ROOT.RooProdPdf("model1", "model1", ROOT.RooArgSet(hc_sig, hc_bkg), Conditional=(model_tmp, x))


#  Case 2 - 'Barlow Beeston' light (one parameter per bin for all samples)

# Construct the histogram shapes, using the same parameters for signal and background
# This requires passing the first histogram to the second, so that their common parameters
# can be re-used.
# The first ParamHistFunc will create one parameter per bin, such as `p_ph_sig2_gamma_bin_0`.
# This allows bin 0 to fluctuate up and down.
# Then, the SAME parameters are connected to the background histogram, so the bins flucutate
# synchronously. This reduces the number of parameters.
p_ph_sig2 = ROOT.RooParamHistFunc("p_ph_sig2", "p_ph_sig2", dh_sig)
p_ph_bkg2 = ROOT.RooParamHistFunc("p_ph_bkg2", "p_ph_bkg2", dh_bkg, p_ph_sig2, True)

Asig2 = ROOT.RooRealVar("Asig", "Asig", 1, 0.01, 5000)
Abkg2 = ROOT.RooRealVar("Abkg", "Abkg", 1, 0.01, 5000)

# As before, construct the sum of signal2 and background2
model2_tmp = ROOT.RooRealSumPdf(
    "sp_ph", "sp_ph", ROOT.RooArgList(p_ph_sig2, p_ph_bkg2), ROOT.RooArgList(Asig2, Abkg2), True
)

# Construct the subsidiary poisson measurements constraining the statistical fluctuations
hc_sigbkg = ROOT.RooHistConstraint("hc_sigbkg", "hc_sigbkg", ROOT.RooArgSet(p_ph_sig2, p_ph_bkg2))

# Construct the joint model
model2 = ROOT.RooProdPdf("model2", "model2", hc_sigbkg, Conditional=(model2_tmp, x))


# ************ Fit all models to data and plot *********************

result0 = model0.fitTo(sumData, PrintLevel=0, Save=True)
result1 = model1.fitTo(sumData, PrintLevel=0, Save=True)
result2 = model2.fitTo(sumData, PrintLevel=0, Save=True)


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
p_ph_sig_set = ROOT.RooArgSet(p_h_sig)
p_ph_bkg_set = ROOT.RooArgSet(p_h_bkg)
model0.plotOn(frame, Components=p_ph_sig_set, LineColor="kAzure")
model0.plotOn(frame, Components=p_ph_bkg_set, LineColor="r")
model0.paramOn(frame)

sigData.plotOn(frame, MarkerColor="b")
frame.Draw()

pt_text1 = [
    "No template uncertainties",
    "are taken into account.",
    "This leads to low errors",
    "for the parameters A, since",
    "the only source of errors",
    "are the data statistics.",
]
for text in pt_text1:
    pt.AddText(text)

pt.DrawClone()


can.cd(2)
frame = x.frame(Title="Barlow Beeston for Sig & Bkg separately")
sumData.plotOn(frame)
model1.plotOn(frame, LineColor="b", VisualizeError=result1)
# Plot data again to show it on top of error bands:
sumData.plotOn(frame)
model1.plotOn(frame, LineColor="b")
p_ph_sig1_set = ROOT.RooArgSet(p_ph_sig1)
p_ph_bkg1_set = ROOT.RooArgSet(p_ph_bkg1)
model1.plotOn(frame, Components=p_ph_sig1_set, LineColor="kAzure")
model1.plotOn(frame, Components=p_ph_bkg1_set, LineColor="r")
model1.paramOn(frame, Parameters=ROOT.RooArgSet(Asig1, Abkg1))

sigData.plotOn(frame, MarkerColor="b")
frame.Draw()

pt.Clear()
pt_text2 = [
    "With gamma parameters, the",
    "signal & background templates",
    "can adapt to the data.",
    "Note how the blue signal",
    "template changes its shape.",
    "This leads to higher errors",
    "of the scale parameters A.",
]

for text in pt_text2:
    pt.AddText(text)

pt.DrawClone()

can.cd(3)
frame = x.frame(Title="Barlow Beeston light for (Sig+Bkg)")
sumData.plotOn(frame)
model2.plotOn(frame, LineColor="b", VisualizeError=result2)
# Plot data again to show it on top of model0 error bands:
sumData.plotOn(frame)
model2.plotOn(frame, LineColor="b")
p_ph_sig2_set = ROOT.RooArgSet(p_ph_sig2)
p_ph_bkg2_set = ROOT.RooArgSet(p_ph_bkg2)
model2.plotOn(frame, Components=p_ph_sig2_set, LineColor="kAzure")
model2.plotOn(frame, Components=p_ph_bkg2_set, LineColor="r")
model2.paramOn(frame, Parameters=ROOT.RooArgSet(Asig2, Abkg2))

sigData.plotOn(frame, MarkerColor="b")
frame.Draw()

pt.Clear()
pt_text3 = [
    "When signal and background",
    "template share one gamma para-",
    "meter per bin, they adapt less.",
    "The errors of the A parameters",
    "also shrink slightly.",
]
for text in pt_text3:
    pt.AddText(text)
pt.DrawClone()


print("Asig [normal ] = {} +/- {}".format(Asig0.getVal(), Asig0.getError()))
print("Asig [BB     ] = {} +/- {}".format(Asig1.getVal(), Asig1.getError()))
print("Asig [BBlight] = {} +/- {}".format(Asig2.getVal(), Asig2.getError()))

can.SaveAs("rf709_BarlowBeeston.png")
