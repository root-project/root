## \file
## \ingroup tutorial_pyroot_legacy
## \notebook
## The ROOT Framework
##
## \macro_image
## \macro_code
##
## \author Wim Lavrijsen

from ROOT import TCanvas, TPavesText
from ROOT import gROOT

c1 = TCanvas("c1","The ROOT Framework",200,10,700,500)
c1.Range(0,0,19,12)

rootf = TPavesText(0.4,0.6,18,2.3,20,"tr")
rootf.AddText("ROOT Framework")
rootf.SetFillColor(42)
rootf.Draw()

eventg = TPavesText(0.99,2.66,3.29,5.67,4,"tr")
eventg.SetFillColor(38)
eventg.AddText("Event")
eventg.AddText("Generators")
eventg.Draw()

simul = TPavesText(3.62,2.71,6.15,7.96,7,"tr")
simul.SetFillColor(41)
simul.AddText("Detector")
simul.AddText("Simulation")
simul.Draw()

recon = TPavesText(6.56,2.69,10.07,10.15,11,"tr")
recon.SetFillColor(48)
recon.AddText("Event")
recon.AddText("Reconstruction")
recon.Draw()

daq = TPavesText(10.43,2.74,14.0,10.81,11,"tr")
daq.AddText("Data")
daq.AddText("Acquisition")
daq.Draw()

anal = TPavesText(14.55,2.72,17.9,10.31,11,"tr")
anal.SetFillColor(42)
anal.AddText("Data")
anal.AddText("Analysis")
anal.Draw()

c1.Update()
