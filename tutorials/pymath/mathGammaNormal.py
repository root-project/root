## \file
## \ingroup tutorial_math
## \notebook
## 
## Tutorial illustrating how to use the TMath::GammaDist and TMath::LogNormal functions.
##
## \macro_image
## \macro_code
##
## \author Anna Kreshuk
## \translator P. P.


import ROOT

#classes
TMath = ROOT.TMath 
TCanvas = ROOT.TCanvas 
TF1 = ROOT.TF1 
TLegend = ROOT.TLegend 

#constant colors
kRed = ROOT.kRed
kMagenta = ROOT.kMagenta
kBlue = ROOT.kBlue
kGreen = ROOT.kGreen



# void
def mathGammaNormal():
   
   global myc
   myc = TCanvas("c1","gamma and lognormal",10,10,600,800)
   myc.Divide(1,2)
   
   global pad1
   pad1 = myc.cd(1) # TPad
   pad1.SetLogy()
   pad1.SetGrid()
   
   #TMath::GammaDist
   global fgamma
   fgamma = TF1("fgamma", "TMath::GammaDist(x, [0], [1], [2])", 0, 10)

   global f1, f2, f3, f4
   fgamma.SetParameters(0.5, 0, 1)
   f1 = fgamma.DrawCopy()
   f1.SetMinimum(1e-5) # Only here we define Minimum, the rest will follow-up.
   f1.SetLineColor(kRed)

   fgamma.SetParameters(1, 0, 1)
   f2 = fgamma.DrawCopy("same")
   f2.SetLineColor(kGreen)

   fgamma.SetParameters(2, 0, 1)
   f3 = fgamma.DrawCopy("same")
   f3.SetLineColor(kBlue)

   fgamma.SetParameters(5, 0, 1)
   f4 = fgamma.DrawCopy("same")
   f4.SetLineColor(kMagenta)

   global legend1
   legend1 = TLegend(.2,.15,.5,.4)
   legend1.AddEntry(f1,"gamma = 0.5 mu = 0  beta = 1","l")
   legend1.AddEntry(f2,"gamma = 1   mu = 0  beta = 1","l")
   legend1.AddEntry(f3,"gamma = 2   mu = 0  beta = 1","l")
   legend1.AddEntry(f4,"gamma = 5   mu = 0  beta = 1","l")
   legend1.Draw()
   
   #TMath::LogNormal
   global pad2
   pad2 = myc.cd(2) # TPad
   pad2.SetLogy()
   pad2.SetGrid()

   #
   global flog
   flog = TF1("flog", "TMath::LogNormal(x, [0], [1], [2])", 0, 5)

   global g1, g2, g3, g4
   flog.SetParameters(0.5, 0, 1)
   g1 = flog.DrawCopy()
   g1.SetLineColor(kRed)

   flog.SetParameters(1, 0, 1)
   g2 = flog.DrawCopy("same")
   g2.SetLineColor(kGreen)

   flog.SetParameters(2, 0, 1)
   g3 = flog.DrawCopy("same")
   g3.SetLineColor(kBlue)

   flog.SetParameters(5, 0, 1)
   g4 = flog.DrawCopy("same")
   g4.SetLineColor(kMagenta)

   global legend2
   legend2 = TLegend(.2,.15,.5,.4)
   legend2.AddEntry(g1,"sigma = 0.5 theta = 0  m = 1","l")
   legend2.AddEntry(g2,"sigma = 1   theta = 0  m = 1","l")
   legend2.AddEntry(g3,"sigma = 2   theta = 0  m = 1","l")
   legend2.AddEntry(g4,"sigma = 5   theta = 0  m = 1","l")
   legend2.Draw()
   



if __name__ == "__main__":
   mathGammaNormal()
