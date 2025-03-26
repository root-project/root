## \file
## \ingroup tutorial_graphs
## \notebook
## \preview Create a polar graph using a TF1 and draw it with PI axis.
## See the [TGraphPolar documentation](https://root.cern/doc/master/classTGraphPolar.html)
##
## Since TGraphPolar is a TGraphErrors, it is painted with
## [TGraphPainter](https://root.cern/doc/master/classTGraphPainter.html) options.
##
## \macro_image
## \macro_code
## \author Olivier Couet, Jamie Gooding

import ROOT
import math
CPol = ROOT.TCanvas("CPol","TGraphPolar Examples",500,500)

rmin = 0
rmax = math.pi*2
r[1000]
theta[1000]

fp1 = ROOT.TF1("fplot","cos(x)",rmin,rmax)
for ipt in range(1000):
   r[ipt] = ipt*(rmax-rmin)/1000+rmin
   theta[ipt] = fp1.Eval(r[ipt])

grP1 = ROOT.TGraphPolar(1000, r, theta)
grP1.SetTitle("")
grP1.SetLineColor(2)
grP1.Draw("AOL")
