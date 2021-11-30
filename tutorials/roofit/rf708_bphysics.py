## \file
## \ingroup tutorial_roofit
## \notebook
## Special pdf's: special decay pdf for B physics with mixing and/or CP violation
##
## \macro_code
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT


# B-decay with mixing
# -------------------------

# Construct pdf
# -------------------------

# Observable
dt = ROOT.RooRealVar("dt", "dt", -10, 10)
dt.setBins(40)

# Parameters
dm = ROOT.RooRealVar("dm", "delta m(B0)", 0.472)
tau = ROOT.RooRealVar("tau", "tau (B0)", 1.547)
w = ROOT.RooRealVar("w", "flavour mistag rate", 0.1)
dw = ROOT.RooRealVar("dw", "delta mistag rate for B0/B0bar", 0.1)

mixState = ROOT.RooCategory("mixState", "B0/B0bar mixing state", {"mixed": -1, "unmixed": 1})

tagFlav = ROOT.RooCategory("tagFlav", "Flavour of the tagged B0", {"B0": 1, "B0bar": -1})

# Use delta function resolution model
tm = ROOT.RooTruthModel("tm", "truth model", dt)

# Construct Bdecay with mixing
bmix = ROOT.RooBMixDecay("bmix", "decay", dt, mixState, tagFlav, tau, dm, w, dw, tm, type="DoubleSided")

# Plot pdf in various slices
# ---------------------------------------------------

# Generate some data
data = bmix.generate({dt, mixState, tagFlav}, 10000)

# Plot B0 and B0bar tagged data separately
# For all plots below B0 and B0 tagged data will look somewhat differently
# if the flavor tagging mistag rate for B0 and B0 is different (i.e. dw!=0)
frame1 = dt.frame(Title="B decay distribution with mixing (B0/B0bar)")

data.plotOn(frame1, Cut="tagFlav==tagFlav::B0")
bmix.plotOn(frame1, Slice=(tagFlav, "B0"))

data.plotOn(frame1, Cut="tagFlav==tagFlav::B0bar", MarkerColor="c")
bmix.plotOn(frame1, Slice=(tagFlav, "B0bar"), LineColor="c")

# Plot mixed slice for B0 and B0bar tagged data separately
frame2 = dt.frame(Title="B decay distribution of mixed events (B0/B0bar)")

data.plotOn(frame2, Cut="mixState==mixState::mixed&&tagFlav==tagFlav::B0")
bmix.plotOn(frame2, Slice={tagFlav: "B0", mixState: "mixed"})

data.plotOn(frame2, Cut="mixState==mixState::mixed&&tagFlav==tagFlav::B0bar", MarkerColor="c")
bmix.plotOn(frame2, Slice={tagFlav: "B0bar", mixState: "mixed"}, LineColor="c")

# Plot unmixed slice for B0 and B0bar tagged data separately
frame3 = dt.frame(Title="B decay distribution of unmixed events (B0/B0bar)")

data.plotOn(frame3, Cut="mixState==mixState::unmixed&&tagFlav==tagFlav::B0")
bmix.plotOn(frame3, Slice={tagFlav: "B0", mixState: "unmixed"})

data.plotOn(frame3, Cut="mixState==mixState::unmixed&&tagFlav==tagFlav::B0bar", MarkerColor="c")
bmix.plotOn(frame3, Slice={tagFlav: "B0bar", mixState: "unmixed"}, LineColor="c")

# B-decay with CP violation
# -------------------------

# Construct pdf
# -------------------------

# Additional parameters needed for B decay with CPV
CPeigen = ROOT.RooRealVar("CPeigen", "CP eigen value", -1)
absLambda = ROOT.RooRealVar("absLambda", "|lambda|", 1, 0, 2)
argLambda = ROOT.RooRealVar("absLambda", "|lambda|", 0.7, -1, 1)
effR = ROOT.RooRealVar("effR", "B0/B0bar reco efficiency ratio", 1)

# Construct Bdecay with CP violation
bcp = ROOT.RooBCPEffDecay(
    "bcp", "bcp", dt, tagFlav, tau, dm, w, CPeigen, absLambda, argLambda, effR, dw, tm, type="DoubleSided"
)

# Plot scenario 1 - sin(2b)=0.7, |l|=1
# ---------------------------------------------------------------------------

# Generate some data
data2 = bcp.generate({dt, tagFlav}, 10000)

# Plot B0 and B0bar tagged data separately
frame4 = dt.frame(Title="B decay distribution with CPV(|l|=1,Im(l)=0.7) (B0/B0bar)")

data2.plotOn(frame4, Cut="tagFlav==tagFlav::B0")
bcp.plotOn(frame4, Slice=(tagFlav, "B0"))

data2.plotOn(frame4, Cut="tagFlav==tagFlav::B0bar", MarkerColor="c")
bcp.plotOn(frame4, Slice=(tagFlav, "B0bar"), LineColor="c")

# # Plot scenario 2 - sin(2b)=0.7, |l|=0.7
# -------------------------------------------------------------------------------

absLambda.setVal(0.7)

# Generate some data
data3 = bcp.generate({dt, tagFlav}, 10000)

# Plot B0 and B0bar tagged data separately (sin2b = 0.7 plus direct CPV
# |l|=0.5)
frame5 = dt.frame(Title="B decay distribution with CPV(|l|=0.7,Im(l)=0.7) (B0/B0bar)")

data3.plotOn(frame5, Cut="tagFlav==tagFlav::B0")
bcp.plotOn(frame5, Slice=(tagFlav, "B0"))

data3.plotOn(frame5, Cut="tagFlav==tagFlav::B0bar", MarkerColor="c")
bcp.plotOn(frame5, Slice=(tagFlav, "B0bar"), LineColor="c")


# Generic B-decay with user coefficients
# -------------------------

# Construct pdf
# -------------------------

# Model parameters
DGbG = ROOT.RooRealVar("DGbG", "DGamma/GammaAvg", 0.5, -1, 1)
Adir = ROOT.RooRealVar("Adir", "-[1-abs(l)**2]/[1+abs(l)**2]", 0)
Amix = ROOT.RooRealVar("Amix", "2Im(l)/[1+abs(l)**2]", 0.7)
Adel = ROOT.RooRealVar("Adel", "2Re(l)/[1+abs(l)**2]", 0.7)

# Derived input parameters for pdf
DG = ROOT.RooFormulaVar("DG", "Delta Gamma", "@1/@0", [tau, DGbG])

# Construct coefficient functions for sin,cos, modulations of decay
# distribution
fsin = ROOT.RooFormulaVar("fsin", "fsin", "@0*@1*(1-2*@2)", [Amix, tagFlav, w])
fcos = ROOT.RooFormulaVar("fcos", "fcos", "@0*@1*(1-2*@2)", [Adir, tagFlav, w])
fsinh = ROOT.RooFormulaVar("fsinh", "fsinh", "@0", [Adel])

# Construct generic B decay pdf using above user coefficients
bcpg = ROOT.RooBDecay(
    "bcpg", "bcpg", dt, tau, DG, ROOT.RooFit.RooConst(1), fsinh, fcos, fsin, dm, tm, type="DoubleSided"
)

# Plot - Im(l)=0.7, e(l)=0.7 |l|=1, G/G=0.5
# -------------------------------------------------------------------------------------

# Generate some data
data4 = bcpg.generate({dt, tagFlav}, 10000)

# Plot B0 and B0bar tagged data separately
frame6 = dt.frame(Title="B decay distribution with CPV(Im(l)=0.7,Re(l)=0.7,|l|=1,dG/G=0.5) (B0/B0bar)")

data4.plotOn(frame6, Cut="tagFlav==tagFlav::B0")
bcpg.plotOn(frame6, Slice=(tagFlav, "B0"))

data4.plotOn(frame6, Cut="tagFlav==tagFlav::B0bar", MarkerColor="c")
bcpg.plotOn(frame6, Slice=(tagFlav, "B0bar"), LineColor="c")

c = ROOT.TCanvas("rf708_bphysics", "rf708_bphysics", 1200, 800)
c.Divide(3, 2)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.6)
frame1.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.6)
frame2.Draw()
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
frame3.GetYaxis().SetTitleOffset(1.6)
frame3.Draw()
c.cd(4)
ROOT.gPad.SetLeftMargin(0.15)
frame4.GetYaxis().SetTitleOffset(1.6)
frame4.Draw()
c.cd(5)
ROOT.gPad.SetLeftMargin(0.15)
frame5.GetYaxis().SetTitleOffset(1.6)
frame5.Draw()
c.cd(6)
ROOT.gPad.SetLeftMargin(0.15)
frame6.GetYaxis().SetTitleOffset(1.6)
frame6.Draw()

c.SaveAs("rf708_bphysics.png")
