## \file
## \ingroup tutorial_graphs
## \notebook
## \preview Draw a scatter plot for 4 variables, mapped to: x, y, marker colour and marker size.
##
## TScatter is available since ROOT v.6.30. See the [TScatter documentation](https://root.cern/doc/master/classTScatter.html)
##
## \macro_image
## \macro_code
## \author Olivier Couet, Jamie Gooding

canvas = ROOT.TCanvas()
canvas.SetRightMargin(0.14)
gStyle.SetPalette(kBird, 0, 0.6) # define a transparent palette

const int n = 175

x = []
y = []
c = []
s = []

# Define four random data sets
r = ROOT.TRandom()
for i in range(n):
   x.append(100*r.Rndm(i))
   y.append(200*r.Rndm(i))
   c.append(300*r.Rndm(i))
   s.append(400*r.Rndm(i))

scatter = ROOT.TScatter(n, x, y, c, s)
scatter.SetMarkerStyle(20)
scatter.SetTitle("Scatter plot titleX titleY titleZ title")
scatter.GetXaxis().SetRangeUser(20.,90.)
scatter.GetYaxis().SetRangeUser(55.,90.)
scatter.GetZaxis().SetRangeUser(10.,200.)
scatter.Draw("A")
