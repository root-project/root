## \file
## \ingroup tutorial_roofit_main
## \notebook -nodraw
## Multidimensional models: performing fits in multiple (disjoint) ranges in one or more dimensions
##
## \macro_code
## \macro_output
##
## \date February 2018
## \authors Clemens Lange, Wouter Verkerke (C++ version)

import ROOT


# Create 2D pdf and data
# -------------------------------------------

# Define observables x,y
x = ROOT.RooRealVar("x", "x", -10, 10)
y = ROOT.RooRealVar("y", "y", -10, 10)

# Construct the signal pdf gauss(x)*gauss(y)
mx = ROOT.RooRealVar("mx", "mx", 1, -10, 10)
my = ROOT.RooRealVar("my", "my", 1, -10, 10)

gx = ROOT.RooGaussian("gx", "gx", x, mx, 1.0)
gy = ROOT.RooGaussian("gy", "gy", y, my, 1.0)

sig = ROOT.RooProdPdf("sig", "sig", gx, gy)

# Construct the background pdf (flat in x,y)
px = ROOT.RooPolynomial("px", "px", x)
py = ROOT.RooPolynomial("py", "py", y)
bkg = ROOT.RooProdPdf("bkg", "bkg", px, py)

# Construct the composite model sig+bkg
f = ROOT.RooRealVar("f", "f", 0.0, 1.0)
model = ROOT.RooAddPdf("model", "model", [sig, bkg], [f])

# Sample 10000 events in (x,y) from the model
modelData = model.generate({x, y}, 10000)

# Define signal and sideband regions
# -------------------------------------------------------------------

# Construct the SideBand1,SideBand2, regions
#
#                    |
#      +-------------+-----------+
#      |             |           |
#      |    Side     |   Sig     |
#      |    Band1    |   nal     |
#      |             |           |
#    --+-------------+-----------+--
#      |                         |
#      |           Side          |
#      |           Band2         |
#      |                         |
#      +-------------+-----------+
#                    |

x.setRange("SB1", -10, +10)
y.setRange("SB1", -10, 0)

x.setRange("SB2", -10, 0)
y.setRange("SB2", 0, +10)

x.setRange("SIG", 0, +10)
y.setRange("SIG", 0, +10)

x.setRange("FULL", -10, +10)
y.setRange("FULL", -10, +10)

# Perform fits in individual sideband regions
# -------------------------------------------------------------------------------------

# Perform fit in SideBand1 region (ROOT.RooAddPdf coefficients will be
# interpreted in full range)
r_sb1 = model.fitTo(modelData, Range="SB1", Save=True, PrintLevel=-1)

# Perform fit in SideBand2 region (ROOT.RooAddPdf coefficients will be
# interpreted in full range)
r_sb2 = model.fitTo(modelData, Range="SB2", Save=True, PrintLevel=-1)

# Perform fits in joint sideband regions
# -----------------------------------------------------------------------------

# Now perform fit to joint 'L-shaped' sideband region 'SB1|SB2'
# (ROOT.RooAddPdf coefficients will be interpreted in full range)
r_sb12 = model.fitTo(modelData, Range="SB1,SB2", Save=True, PrintLevel=-1)

# Print results for comparison
r_sb1.Print()
r_sb2.Print()
r_sb12.Print()
