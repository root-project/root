## \ingroup tutorial_roofit
## \notebook
##
## 'VALIDATION AND MC STUDIES' RooFit tutorial macro #804
##
## Using RooMCStudy on models with constraints
##
## \macro_code
##
## \date February 2018
## \author Clemens Lange


import ROOT


# Create model with parameter constraint
# ---------------------------------------------------------------------------

# Observable
x = ROOT.RooRealVar("x", "x", -10, 10)

# Signal component
m = ROOT.RooRealVar("m", "m", 0, -10, 10)
s = ROOT.RooRealVar("s", "s", 2, 0.1, 10)
g = ROOT.RooGaussian("g", "g", x, m, s)

# Background component
p = ROOT.RooPolynomial("p", "p", x)

# Composite model
f = ROOT.RooRealVar("f", "f", 0.4, 0.0, 1.0)
sum = ROOT.RooAddPdf("sum", "sum", [g, p], [f])

# Construct constraint on parameter f
fconstraint = ROOT.RooGaussian("fconstraint", "fconstraint", f, ROOT.RooFit.RooConst(0.7), ROOT.RooFit.RooConst(0.1))

# Multiply constraint with p.d.f
sumc = ROOT.RooProdPdf("sumc", "sum with constraint", [sum, fconstraint])

# Setup toy study with model
# ---------------------------------------------------

# Perform toy study with internal constraint on f
mcs = ROOT.RooMCStudy(
    sumc,
    {x},
    ROOT.RooFit.Constrain({f}),
    ROOT.RooFit.Silence(),
    ROOT.RooFit.Binned(),
    ROOT.RooFit.FitOptions(ROOT.RooFit.PrintLevel(-1)),
)

# Run 500 toys of 2000 events.
# Before each toy is generated, value for the f is sampled from the constraint pdf and
# that value is used for the generation of that toy.
mcs.generateAndFit(500, 2000)

# Make plot of distribution of generated value of f parameter
h_f_gen = mcs.fitParDataSet().createHistogram("f_gen", ROOT.RooFit.AutoBinning(40))

# Make plot of distribution of fitted value of f parameter
frame1 = mcs.plotParam(f, ROOT.RooFit.Bins(40))
frame1.SetTitle("Distribution of fitted f values")

# Make plot of pull distribution on f
frame2 = mcs.plotPull(f, ROOT.RooFit.Bins(40), ROOT.RooFit.FitGauss())
frame1.SetTitle("Distribution of f pull values")

c = ROOT.TCanvas("rf804_mcstudy_constr", "rf804_mcstudy_constr", 1200, 400)
c.Divide(3)
c.cd(1)
ROOT.gPad.SetLeftMargin(0.15)
h_f_gen.GetYaxis().SetTitleOffset(1.4)
h_f_gen.Draw()
c.cd(2)
ROOT.gPad.SetLeftMargin(0.15)
frame1.GetYaxis().SetTitleOffset(1.4)
frame1.Draw()
c.cd(3)
ROOT.gPad.SetLeftMargin(0.15)
frame2.GetYaxis().SetTitleOffset(1.4)
frame2.Draw()

c.SaveAs("rf804_mcstudy_constr.png")
