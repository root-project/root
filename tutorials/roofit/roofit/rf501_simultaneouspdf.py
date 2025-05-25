## \file
## \ingroup tutorial_roofit_main
## \notebook
## Organization and simultaneous fits: using simultaneous pdfs to describe simultaneous
## fits to multiple datasets
##
## \macro_image
## \macro_code
## \macro_output
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT


# Create model for physics sample
# -------------------------------------------------------------

# Create observables
x = ROOT.RooRealVar("x", "x", -8, 8)

# Construct signal pdf
mean = ROOT.RooRealVar("mean", "mean", 0, -8, 8)
sigma = ROOT.RooRealVar("sigma", "sigma", 0.3, 0.1, 10)
gx = ROOT.RooGaussian("gx", "gx", x, mean, sigma)

# Construct background pdf
a0 = ROOT.RooRealVar("a0", "a0", -0.1, -1, 1)
a1 = ROOT.RooRealVar("a1", "a1", 0.004, -1, 1)
px = ROOT.RooChebychev("px", "px", x, [a0, a1])

# Construct composite pdf
f = ROOT.RooRealVar("f", "f", 0.2, 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [gx, px], [f])

# Create model for control sample
# --------------------------------------------------------------

# Construct signal pdf.
# NOTE that sigma is shared with the signal sample model
mean_ctl = ROOT.RooRealVar("mean_ctl", "mean_ctl", -3, -8, 8)
gx_ctl = ROOT.RooGaussian("gx_ctl", "gx_ctl", x, mean_ctl, sigma)

# Construct the background pdf
a0_ctl = ROOT.RooRealVar("a0_ctl", "a0_ctl", -0.1, -1, 1)
a1_ctl = ROOT.RooRealVar("a1_ctl", "a1_ctl", 0.5, -0.1, 1)
px_ctl = ROOT.RooChebychev("px_ctl", "px_ctl", x, [a0_ctl, a1_ctl])

# Construct the composite model
f_ctl = ROOT.RooRealVar("f_ctl", "f_ctl", 0.5, 0.0, 1.0)
model_ctl = ROOT.RooAddPdf("model_ctl", "model_ctl", [gx_ctl, px_ctl], [f_ctl])

# Generate events for both samples
# ---------------------------------------------------------------

# Generate 1000 events in x and y from model
data = model.generate({x}, 1000)
data_ctl = model_ctl.generate({x}, 2000)

# Create index category and join samples
# ---------------------------------------------------------------------------

# Define category to distinguish physics and control samples events
sample = ROOT.RooCategory("sample", "sample")
sample.defineType("physics")
sample.defineType("control")

# Construct combined dataset in (x,sample)
combData = ROOT.RooDataSet(
    "combData",
    "combined data",
    {x},
    Index=sample,
    Import={"physics": data, "control": data_ctl},
)

# Construct a simultaneous pdf in (x, sample)
# -----------------------------------------------------------------------------------

# Construct a simultaneous pdf using category sample as index: associate model
# with the physics state and model_ctl with the control state
simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", {"physics": model, "control": model_ctl}, sample)

# Perform a simultaneous fit
# ---------------------------------------------------

# Perform simultaneous fit of model to data and model_ctl to data_ctl
fitResult = simPdf.fitTo(combData, PrintLevel=-1, Save=True)
fitResult.Print()

# Plot model slices on data slices
# ----------------------------------------------------------------

# Make a frame for the physics sample
frame1 = x.frame(Title="Physics sample")

# Plot all data tagged as physics sample
combData.plotOn(frame1, Cut="sample==sample::physics")

# Plot "physics" slice of simultaneous pdf.
# NB: You *must* project the sample index category with data using ProjWData as
# a RooSimultaneous makes no prediction on the shape in the index category and
# can thus not be integrated. In other words: Since the PDF doesn't know the
# number of events in the different category states, it doesn't know how much
# of each component it has to project out. This info is read from the data.
simPdf.plotOn(frame1, Slice=(sample, "physics"), ProjWData=(sample, combData))
simPdf.plotOn(frame1, Slice=(sample, "physics"), Components="px", ProjWData=(sample, combData), LineStyle="--")

# The same plot for the control sample slice. We do this with a different
# approach this time, for illustration purposes. Here, we are slicing the
# dataset and then use the data slice for the projection, because then the
# RooFit::Slice() becomes unnecessary. This approach is more general,
# because you can plot sums of slices by using logical or in the Cut()
# command.
frame2 = x.frame(Title="Control sample")
slicedData = combData.reduce(Cut="sample==sample::control")
slicedData.plotOn(frame2)
simPdf.plotOn(frame2, ProjWData=(sample, slicedData))
simPdf.plotOn(frame2, Components="px_ctl", ProjWData=(sample, slicedData), LineStyle="--")

# The same plot for all the phase space. Here, we can just use the original
# combined dataset.
frame3 = x.frame(Title="Both samples")
combData.plotOn(frame3)
simPdf.plotOn(frame3, ProjWData=(sample, combData))
simPdf.plotOn(frame3, Components="px,px_ctl", ProjWData=(sample, combData), LineStyle="--")

c = ROOT.TCanvas("rf501_simultaneouspdf", "rf501_simultaneouspdf", 1200, 400)
c.Divide(3)


def draw(i, frame):
    c.cd(i)
    ROOT.gPad.SetLeftMargin(0.15)
    frame.GetYaxis().SetTitleOffset(1.4)
    frame.Draw()


draw(1, frame1)
draw(2, frame2)
draw(3, frame3)

c.SaveAs("rf501_simultaneouspdf.png")
