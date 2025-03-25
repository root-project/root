## \file
## \ingroup tutorial_graphs
## \notebook -js
## Graph with bent error bars. Inspired from work of Olivier Couet.
##
## See the [TGraphBentErrors documentation](https://root.cern/doc/master/classTGraphBentErrors.html)
##
## exl / exh: low and high (left/right) errors in x; similar for y
## e*d: delta, in axis units, to be added/subtracted (if >0 or <0) in x or y from
## the data point's position to use as end point of the corresponding error
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro

import ROOT

c1 = ROOT.TCanvas()
n = 10
x = ROOT.std.vector("double")()  # The equivalent is also achieveable with numpy arrays
for i in [-0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95]:
    x.push_back(i)
y = ROOT.std.vector("double")()
for i in [1, 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1]:
    y.push_back(i)
exl = ROOT.std.vector("double")()
for i in [0.05, 0.1, 0.07, 0.07, 0.04, 0.05, 0.06, 0.07, 0.08, 0.05]:
    exl.push_back(i)
eyl = ROOT.std.vector("double")()
for i in [0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8]:
    eyl.push_back(i)
exh = ROOT.std.vector("double")()
for i in [0.02, 0.08, 0.05, 0.05, 0.03, 0.03, 0.04, 0.05, 0.06, 0.03]:
    exh.push_back(i)
eyh = ROOT.std.vector("double")()
for i in [0.6, 0.5, 0.4, 0.3, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6]:
    eyh.push_back(i)
exld = ROOT.std.vector("double")()
for i in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
    exld.push_back(i)
eyld = ROOT.std.vector("double")()
for i in [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
    eyld.push_back(i)
exhd = ROOT.std.vector("double")()
for i in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
    exhd.push_back(i)
eyhd = ROOT.std.vector("double")()
for i in [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0]:
    eyhd.push_back(i)

gr = ROOT.TGraphBentErrors(
    n,
    x.data(),
    y.data(),
    exl.data(),
    exh.data(),
    eyl.data(),
    eyh.data(),
    exld.data(),
    exhd.data(),
    eyld.data(),
    eyhd.data(),
)

gr.SetTitle("TGraphBentErrors Example")
gr.SetMarkerColor(4)
gr.SetMarkerStyle(21)
gr.Draw("ALP")
