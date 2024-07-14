## \file
## \ingroup tutorial_graphics
## \notebook
##
## An example with basic graphics illustrating the Object Oriented User Interface of ROOT.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TPavesText = ROOT.TPavesText

# void
def framework() :

   #
   global c1
   c1 = TCanvas("c1","The ROOT Framework",200,10,700,500)
   c1.Range(0,0,19,12)
   
   #
   global rootf
   rootf = TPavesText(0.4,0.6,18,2.3,20,"tr")
   rootf.AddText("ROOT Framework")
   rootf.SetFillColor(42)
   rootf.Draw()
   
   #
   global eventg
   eventg = TPavesText(0.99,2.66,3.29,5.67,4,"tr")
   eventg.SetFillColor(38)
   eventg.AddText("Event")
   eventg.AddText("Generators")
   eventg.Draw()
   
   #
   global simul
   simul = TPavesText(3.62,2.71,6.15,7.96,7,"tr")
   simul.SetFillColor(41)
   simul.AddText("Detector")
   simul.AddText("Simulation")
   simul.Draw()
   
   #
   global recon
   recon = TPavesText(6.56,2.69,10.07,10.15,11,"tr")
   recon.SetFillColor(48)
   recon.AddText("Event")
   recon.AddText("Reconstruction")
   recon.Draw()
   
   #
   global daq
   daq = TPavesText(10.43,2.74,14.0,10.81,11,"tr")
   daq.AddText("Data")
   daq.AddText("Acquisition")
   daq.Draw()
   
   #
   global analysis
   analysis = TPavesText(14.55,2.72,17.9,10.31,11,"tr")
   analysis.SetFillColor(42)
   analysis.AddText("Data")
   analysis.AddText("Analysis")
   analysis.Draw()

   c1.Update()
   


if __name__ == "__main__":
   framework()
