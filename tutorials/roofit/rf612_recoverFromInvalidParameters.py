## \file
## \ingroup tutorial_roofit
## \notebook
## Likelihood and minimization: Recover from regions where the function is not defined.
##
## We demonstrate improved recovery from disallowed parameters. For this, we use a polynomial PDF of the form
## \f[
##   \mathrm{Pol2} = \mathcal{N} \left( c + a_1 \cdot x + a_2 \cdot x^2 + 0.01 \cdot x^3 \right),
## \f]
## where \f$ \mathcal{N} \f$ is a normalisation factor. Unless the parameters are chosen carefully,
## this function can be negative, and hence, it cannot be used as a PDF. In this case, RooFit passes
## an error to the minimiser, which might try to recover.
##
## \macro_output
## \macro_code
##
## \date June 2021
## \author Harshal Shende, Stephan Hageboeck (C++ version)

import ROOT


# Create a fit model:
# The polynomial is notoriously unstable, because it can quickly go negative.
# Since PDFs need to be positive, one often ends up with an unstable fit model.
x = ROOT.RooRealVar("x", "x", -15, 15)
a1 = ROOT.RooRealVar("a1", "a1", -0.5, -10.0, 20.0)
a2 = ROOT.RooRealVar("a2", "a2", 0.2, -10.0, 20.0)
a3 = ROOT.RooRealVar("a3", "a3", 0.01)
pdf = ROOT.RooPolynomial("pol3", "c + a1 * x + a2 * x*x + 0.01 * x*x*x", x, [a1, a2, a3])

# Create toy data with all-positive coefficients:
data = pdf.generate(x, 10000)

# For plotting.
# We create pointers to the plotted objects. We want these objects to leak out of the function,
# so we can still see them after it returns.
c = ROOT.TCanvas()
frame = x.frame()
data.plotOn(frame, Name="data")

# Plotting a PDF with disallowed parameters doesn't work. We would get a lot of error messages.
# Therefore, we disable plotting messages in RooFit's message streams:
ROOT.RooMsgService.instance().getStream(0).removeTopic(ROOT.RooFit.Plotting)
ROOT.RooMsgService.instance().getStream(1).removeTopic(ROOT.RooFit.Plotting)


# RooFit before ROOT 6.24
# --------------------------------
# Before 6.24, RooFit wasn't able to recover from invalid parameters. The minimiser just errs around
# the starting values of the parameters without finding any improvement.

# Set up the parameters such that the PDF would come out negative. The PDF is now undefined.
a1.setVal(10.0)
a2.setVal(-1.0)

# Perform a fit:
fitWithoutRecovery = pdf.fitTo(
    data,
    Save=True,
    RecoverFromUndefinedRegions=0.0,  # This is how RooFit behaved prior to ROOT 6.24
    PrintEvalErrors=-1,  # We are expecting a lot of evaluation errors. -1 switches off printing.
    PrintLevel=-1,
)

pdf.plotOn(frame, LineColor="r", Name="noRecovery")


# RooFit since ROOT 6.24
# --------------------------------
# The minimiser gets information about the "badness" of the violation of the function definition. It uses this
# to find its way out of the disallowed parameter regions.
print("\n\n\n-------------- Starting second fit ---------------\n\n")

# Reset the parameters such that the PDF is again undefined.
a1.setVal(10.0)
a2.setVal(-1.0)

# Fit again, but pass recovery information to the minimiser:
fitWithRecovery = pdf.fitTo(
    data,
    Save=True,
    RecoverFromUndefinedRegions=1.0,  # The magnitude of the recovery information can be chosen here.
    # Higher values mean more aggressive recovery.
    PrintEvalErrors=-1,  # We are still expecting a few evaluation errors.
    PrintLevel=0,
)

pdf.plotOn(frame, LineColor="b", Name="recovery")


# Collect results and plot.
# --------------------------------
# We print the two fit results, and plot the fitted curves.
# The curve of the fit without recovery cannot be plotted, because the PDF is undefined if a2 < 0.
fitWithoutRecovery.Print()
print(
    "Without recovery, the fitter encountered {}".format(fitWithoutRecovery.numInvalidNLL())
    + " invalid function values. The parameters are unchanged.\n"
)

fitWithRecovery.Print()
print(
    "With recovery, the fitter encountered {}".format(fitWithoutRecovery.numInvalidNLL())
    + " invalid function values, but the parameters are fitted.\n"
)

legend = ROOT.TLegend(0.5, 0.7, 0.9, 0.9)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
legend.AddEntry(frame.findObject("data"), "Data", "P")
legend.AddEntry(frame.findObject("noRecovery"), "Without recovery (cannot be plotted)", "L")
legend.AddEntry(frame.findObject("recovery"), "With recovery", "L")
frame.Draw()
legend.Draw()
c.Draw()

c.SaveAs("rf612_recoverFromInvalidParameters.png")
