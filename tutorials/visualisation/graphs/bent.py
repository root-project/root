## \file
## \ingroup tutorial_graphs
## \notebook -js
## Bent error bars. Inspired from work of Olivier Couet.
##
## \macro_image
## \macro_code
##
## \author Alberto Ferro

import ROOT

c1 = ROOT.TCanvas()
n = 10
x = ROOT.std.vector('double')()
for i in [-0.22, 0.05, 0.25, 0.35, 0.5, 0.61,0.7,0.85,0.89,0.95]: x.push_back(i)
y = ROOT.std.vector('double')()
for i in [1,2.9,5.6,7.4,9,9.6,8.7,6.3,4.5,1]: y.push_back(i)
exl = ROOT.std.vector('double')()
for i in [.05,.1,.07,.07,.04,.05,.06,.07,.08,.05]: exl.push_back(i)
eyl = ROOT.std.vector('double')()
for i in [.8,.7,.6,.5,.4,.4,.5,.6,.7,.8]: eyl.push_back(i)
exh = ROOT.std.vector('double')()
for i in [.02,.08,.05,.05,.03,.03,.04,.05,.06,.03]: exh.push_back(i)
eyh = ROOT.std.vector('double')()
for i in [.6,.5,.4,.3,.2,.2,.3,.4,.5,.6]: eyh.push_back(i)
exld = ROOT.std.vector('double')()
for i in [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]: exld.push_back(i)
eyld = ROOT.std.vector('double')()
for i in [.0,.0,.05,.0,.0,.0,.0,.0,.0,.0]: eyld.push_back(i)
exhd = ROOT.std.vector('double')()
for i in [.0,.0,.0,.0,.0,.0,.0,.0,.0,.0]: exhd.push_back(i)
eyhd = ROOT.std.vector('double')()
for i in [.0,.0,.0,.0,.0,.0,.0,.0,.05,.0]: eyhd.push_back(i)

gr = ROOT.TGraphBentErrors(
   n,x.data(),y.data(),exl.data(),exh.data(),eyl.data(),eyh.data(),exld.data(),exhd.data(),eyld.data(),eyhd.data())

gr.SetTitle("TGraphBentErrors Example")
gr.SetMarkerColor(4)
gr.SetMarkerStyle(21)
gr.Draw("ALP")
