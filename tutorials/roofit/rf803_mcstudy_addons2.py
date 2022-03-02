## \ingroup tutorial_roofit
## \notebook
##
## 'VALIDATION AND MC STUDIES' RooFit tutorial macro #803
##
## RooMCStudy: Using the randomizer and profile likelihood add-on models
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange


import ROOT


# Create model
# -----------------------

# Simulation of signal and background of top quark decaying into
# 3 jets with background

# Observable
mjjj = ROOT.RooRealVar("mjjj", "m(3jet) (GeV)", 100, 85.0, 350.0)

# Signal component (Gaussian)
mtop = ROOT.RooRealVar("mtop", "m(top)", 162)
wtop = ROOT.RooRealVar("wtop", "m(top) resolution", 15.2)
sig = ROOT.RooGaussian("sig", "top signal", mjjj, mtop, wtop)

# Background component (Chebychev)
c0 = ROOT.RooRealVar("c0", "Chebychev coefficient 0", -0.846, -1.0, 1.0)
c1 = ROOT.RooRealVar("c1", "Chebychev coefficient 1", 0.112, -1.0, 1.0)
c2 = ROOT.RooRealVar("c2", "Chebychev coefficient 2", 0.076, -1.0, 1.0)
bkg = ROOT.RooChebychev("bkg", "combinatorial background", mjjj, [c0, c1, c2])

# Composite model
nsig = ROOT.RooRealVar("nsig", "number of signal events", 53, 0, 1e3)
nbkg = ROOT.RooRealVar("nbkg", "number of background events", 103, 0, 5e3)
model = ROOT.RooAddPdf("model", "model", [sig, bkg], [nsig, nbkg])

# Create manager
# ---------------------------

# Configure manager to perform binned extended likelihood fits (Binned(), ROOT.RooFit.Extended()) on data generated
# with a Poisson fluctuation on Nobs (Extended())
mcs = ROOT.RooMCStudy(
    model,
    {mjjj},
    ROOT.RooFit.Binned(),
    ROOT.RooFit.Silence(),
    ROOT.RooFit.Extended(ROOT.kTRUE),
    ROOT.RooFit.FitOptions(ROOT.RooFit.Extended(ROOT.kTRUE), ROOT.RooFit.PrintEvalErrors(-1)),
)

# Customize manager
# ---------------------------------

# Add module that randomizes the summed value of nsig+nbkg
# sampling from a uniform distribution between 0 and 1000
#
# In general one can randomize a single parameter, a
# sum of N parameters, either a uniform or a Gaussian
# distribution. Multiple randomization can be executed
# by a single randomizer module

randModule = ROOT.RooRandomizeParamMCSModule()
randModule.sampleSumUniform({nsig, nbkg}, 50, 500)
mcs.addModule(randModule)

# Add profile likelihood calculation of significance. Redo each
# fit while keeping parameter nsig fixed to zero. For each toy,
# the difference in -log(L) of both fits is stored, well
# a simple significance interpretation of the delta(-logL)
# Dnll = 0.5 sigma^2

sigModule = ROOT.RooDLLSignificanceMCSModule(nsig, 0)
mcs.addModule(sigModule)

# Run manager, make plots
# ---------------------------------------------

# Run 1000 experiments. ROOT.This configuration will generate a fair number
# of (harmless) MINUIT warnings due to the instability of the Chebychev polynomial fit
# at low statistics.
mcs.generateAndFit(500)

# Make some plots
dll_vs_ngen = ROOT.RooAbsData.createHistogram(mcs.fitParDataSet(), "ngen,dll_nullhypo_nsig", -40, -40)
z_vs_ngen = ROOT.RooAbsData.createHistogram(mcs.fitParDataSet(), "ngen,significance_nullhypo_nsig", -40, -40)
errnsig_vs_ngen = ROOT.RooAbsData.createHistogram(mcs.fitParDataSet(), "ngen,nsigerr", -40, -40)
errnsig_vs_nsig = ROOT.RooAbsData.createHistogram(mcs.fitParDataSet(), "nsig,nsigerr", -40, -40)

# Draw plots on canvas
c = ROOT.TCanvas("rf803_mcstudy_addons2", "rf802_mcstudy_addons2", 800, 800)
c.Divide(2, 2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
dll_vs_ngen.GetYaxis().SetTitleOffset(1.6)
dll_vs_ngen.Draw("box")
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
z_vs_ngen.GetYaxis().SetTitleOffset(1.6)
z_vs_ngen.Draw("box")
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
errnsig_vs_ngen.GetYaxis().SetTitleOffset(1.6)
errnsig_vs_ngen.Draw("box")
c.cd(4)
ROOT.gPad.SetLeftMargin(0.15)
errnsig_vs_nsig.GetYaxis().SetTitleOffset(1.6)
errnsig_vs_nsig.Draw("box")

c.SaveAs("rf803_mcstudy_addons2.png")

# Make ROOT.RooMCStudy object available on command line after
# macro finishes
ROOT.gDirectory.Add(mcs)
