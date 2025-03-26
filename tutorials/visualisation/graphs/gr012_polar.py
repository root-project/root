## \file
## \ingroup tutorial_graphs
## \notebook
## \preview Create and draw a polar graph. See the [TGraphPolar documentation](https://root.cern/doc/master/classTGraphPolar.html)
##
## Since TGraphPolar is a TGraphErrors, it is painted with
## [TGraphPainter](https://root.cern/doc/master/classTGraphPainter.html) options.
##
## With GetPolargram we retrieve the polar axis to format it; see the
## [TGraphPolargram documentation](https://root.cern/doc/master/classTGraphPolargram.html)
##
## \macro_image
## \macro_code
## \author Olivier Couet, Jamie Gooding

# Illustrates how to use TGraphPolar

import ROOT
import math

CPol = ROOT.TCanvas("CPol","TGraphPolar Examples",1200,600)
CPol.Divide(2,1)

# Left-side pad. Two graphs without errors
CPol.cd(1)
xmin=0
xmax=math.pi*2

x = []
y = []
xval1 = []
yval1 = []

# Graph 1 to be drawn with line and fill
fplot = ROOT.TF1("fplot","cos(2*x)*cos(20*x)",xmin,xmax)
for ipt in range(1000):
   x.append(ipt*(xmax-xmin)/1000 + xmin)
   y.append(fplot.Eval(x[ipt]))

grP = ROOT.TGraphPolar(1000,x,y)
grP.SetLineColor(2)
grP.SetLineWidth(2)
grP.SetFillStyle(3012)
grP.SetFillColor(2)
grP.Draw("AFL")

# Graph 2 to be drawn superposed over graph 1, with curve and polymarker
for ipt in range(20):
   xval1.append(x[1000/20*ipt])
   yval1.append(y[1000/20*ipt])

grP1 = ROOT.TGraphPolar(20,xval1,yval1)
grP1.SetMarkerStyle(29)
grP1.SetMarkerSize(2)
grP1.SetMarkerColor(4)
grP1.SetLineColor(4)
grP1.Draw("CP")

# To format the polar axis, we retrieve the TGraphPolargram.
# First update the canvas, otherwise GetPolargram returns 0
CPol.Update()
if grP1.GetPolargram():
   grP1.GetPolargram().SetTextColor(8)
   grP1.GetPolargram().SetRangePolar(-math.pi,math.pi)
   grP1.GetPolargram().SetNdivPolar(703)
   grP1.GetPolargram().SetToRadian() # tell ROOT that the x and xval1 are in radians


# Right-side pad. One graph with errors
CPol.cd(2)
x2 = []
y2 = []
ex = []
ey = []
for ipt in range(30):
   x2.append(math.floor(x[1000/30*ipt]))
   y2.append(1.2 + 0.4*math.sin(math.pi*2*ipt/30))
   ex.append(0.2+0.1*math.cos(2*math.pi/30*ipt))
   ey.append(0.2)

# Grah to be drawn with polymarker and errors
grPE = ROOT.TGraphPolar(30,x2,y2,ex,ey)
grPE.SetMarkerStyle(22)
grPE.SetMarkerSize(1.5)
grPE.SetMarkerColor(5)
grPE.SetLineColor(6)
grPE.SetLineWidth(2)
grPE.Draw("EP")

# To format the polar axis, we retrieve the TGraphPolargram.
# First update the canvas, otherwise GetPolargram returns 0
CPol.Update()
if grPE.GetPolargram():
   grPE.GetPolargram().SetTextSize(0.03)
   grPE.GetPolargram().SetTwoPi()
   grPE.GetPolargram().SetToRadian() # tell ROOT that the x2 values are in radians

