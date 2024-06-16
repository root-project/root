# \file
# \ingroup tutorial_physics
# \notebook -js
# Example of use of TGenPhaseSpace
#
# \macro_image
# \macro_code
#
# \author Valerio Filippini
# \translator P. P.

import ROOT
import ctypes

TH2F = ROOT.TH2F
TCanvas = ROOT.TCanvas

TLorentzVector = ROOT.TLorentzVector
TGenPhaseSpace = ROOT.TGenPhaseSpace

c_double = ctypes.c_double

def PhaseSpace():
   
   target = TLorentzVector(0.0, 0.0, 0.0, 0.938)
   beam = TLorentzVector(0.0, 0.0, .65, .65)
   W = beam + target
   
   #(Momentum, Energy units are Gev/C, GeV)
   masses = [ 0.938, 0.139, 0.139 ]
   c_masses = (c_double * 3)(*masses)
   
   event = TGenPhaseSpace()
   event.SetDecay(W, 3, c_masses)
   
   h2 =  TH2F("h2","h2", 50,1.1,1.8, 50,1.1,1.8)
   
   #TODO: Get better conventiion for names variables for pPPip, pPim, ...
   for n in range(100000):
      weight = event.Generate()
      
      pProton = event.GetDecay(0)
      
      pPip = event.GetDecay(1)
      pPim = event.GetDecay(2)
      
      pPPip = pProton + pPip
      pPPim = pProton + pPim
      
      h2.Fill(pPPip.M2() ,pPPim.M2() ,weight)
      
   myc = TCanvas("myc", "myc")
   h2.Draw()
   myc.Update()
   myc.SaveAs("PhaseSpace.png")
   
if __name__ == "__main__":
   PhaseSpace()
