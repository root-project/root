## \file
## \ingroup tutorial_roofit_main
## \notebook
## Use Morphing in RooFit.
##
## This tutorial shows how to use template morphing inside RooFit. As input we have several
## Gaussian distributions. The output is one gaussian, with a specific mean value.
## Since likelihoods are often used within the framework of morphing, we provide a
## way to estimate the negative log likelihood (nll).
##
## Based on example of Kyle Cranmer https://gist.github.com/cranmer/46fff8d22015e5a26619.
##
## \macro_image
## \macro_code
## \macro_output
##
## \date August 2024
## \author Robin Syring


import ROOT

# Number of samples to fill the histograms
n_samples = 1000


# Kills warning messages
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)


# morphing as a baseline
def morphing(setting):
    # set up a frame for plotting
    frame1 = x_var.frame()

    # define binning for morphing
    bin_mu_x = ROOT.RooBinning(4, 0.0, 4.0)
    grid = ROOT.RooMomentMorphFuncND.Grid(bin_mu_x)
    x_var.setBins(50)

    # number of 'sampled' Gaussians, if you change it, adjust the binning properly
    for i in range(5):
        # Create the sampled Gaussian
        workspace.factory(f"Gaussian::g{i}(x, mu{i}[{i}], {sigma})".format(i=i))

        # Fill the histograms
        hist = workspace[f"g{i}"].generateBinned([x_var], n_samples * 100)
        # Make sure that every bin is filled and we don't get zero probability
        for i_bin in range(hist.numEntries()):
            hist.add(hist.get(i_bin), 1.0)

        # Add the pdf to the workspace, the inOrder of 1 is necessary for calculation of the nll
        # Adjust it to 0 to see binning
        workspace.Import(ROOT.RooHistPdf(f"histpdf{i}", f"histpdf{i}", [x_var], hist, intOrder=1))

        # Add the pdf to the grid and to the plot
        grid.addPdf(workspace[f"histpdf{i}"], int(i))
        workspace[f"histpdf{i}"].plotOn(frame1)

    # Create the morphing and add it to the workspace
    morph_func = ROOT.RooMomentMorphFuncND("morph_func", "morph_func", [mu_var], [x_var], grid, setting)

    # Normalizes the morphed object to be a pdf, set it false to prevent warning messages and gain computational speed up
    morph_func.setPdfMode()

    # Creating the morphed pdf
    morph = ROOT.RooWrapperPdf("morph", "morph", morph_func, True)
    workspace.Import(morph)
    workspace["morph"].plotOn(frame1, LineColor="r")

    return frame1


# Define the "observed" data in a workspade
def build_ws(mu_observed, sigma):
    ws = ROOT.RooWorkspace()
    ws.factory(f"Gaussian::gauss(x[-5,15], mu[{mu_observed},0,4], {sigma})".format(mu_observed=mu_observed))

    return ws


# The "observed" data
mu_observed = 2.5
sigma = 1.5
workspace = build_ws(mu_observed, sigma)
x_var = workspace["x"]
mu_var = workspace["mu"]
gauss = workspace.pdf("gauss")
obs_data = gauss.generate(x_var, n_samples)


# Create the exact negative log likelihood functions for Gaussian model
nll_gauss = gauss.createNLL(obs_data)

# Compute the morphed nll
frame1 = morphing(ROOT.RooMomentMorphFuncND.Linear)

# TODO: Fix RooAddPdf::fixCoefNormalization(nset) warnings with new CPU backend
nll_morph = workspace["morph"].createNLL(obs_data, EvalBackend="legacy")

# Plot the negative logarithmic summed likelihood
frame2 = mu_var.frame(Title="Negative log Likelihood")
nll_gauss.plotOn(frame2, LineColor="b", ShiftToZero=True, Name="gauss")
nll_morph.plotOn(frame2, LineColor="r", ShiftToZero=True, Name="morphed")

c = ROOT.TCanvas("rf616_morphing", "rf616_morphing", 800, 400)
c.Divide(2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.8)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.8)
frame2.Draw()


# Compute the minimum via minuit and display the results
for nll in [nll_gauss, nll_morph]:
    minimizer = ROOT.RooMinimizer(nll)
    minimizer.setPrintLevel(-1)
    minimizer.minimize("Minuit2")
    result = minimizer.save()
    result.Print()
