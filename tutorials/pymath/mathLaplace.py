## \file
## \ingroup tutorial_math
## \notebook
##
## Test the TMath::LaplaceDist and TMath::LaplaceDistI functions of ROOT.
##
## \macro_image
## \macro_code
##
## \author Anna Kreshuk
## \translator P. P.


import ROOT

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
def mathLaplace():
   
   global c1
   c1 = TCanvas("c1", "TMath::LaplaceDist",600,800)
   c1.Divide(1, 2)
   
   global pad1
   pad1 = c1.cd(1)
   pad1.SetGrid()

   global flaplace
   flaplace = TF1("flaplace", "TMath::LaplaceDist(x, [0], [1])", -10, 10)

   global f1, f2, f3, f4
   flaplace.SetParameters(0, 1)
   f1 = flaplace.DrawCopy()
   f1.SetLineColor(kRed)
   f1.SetLineWidth(1)

   flaplace.SetParameters(0, 2)
   f2 = flaplace.DrawCopy("same")
   f2.SetLineColor(kGreen)
   f2.SetLineWidth(1)

   flaplace.SetParameters(2, 1)
   f3 = flaplace.DrawCopy("same")
   f3.SetLineColor(kBlue)
   f3.SetLineWidth(1)

   flaplace.SetParameters(2, 2)
   f4 = flaplace.DrawCopy("same")
   f4.SetLineColor(kMagenta)
   f4.SetLineWidth(1)
   
   global legend1
   legend1 = TLegend(.7,.7,.9,.9)
   legend1.AddEntry(f1,"alpha=0 beta=1","l")
   legend1.AddEntry(f2,"alpha=0 beta=2","l")
   legend1.AddEntry(f3,"alpha=2 beta=1","l")
   legend1.AddEntry(f4,"alpha=2 beta=2","l")
   legend1.Draw()
   
   global pad2
   pad2 = c1.cd(2)
   pad2.SetGrid()
   
   global flaplacei
   flaplacei = TF1("flaplacei", "TMath::LaplaceDistI(x, [0], [1])", -10, 10)

   global g1, g2, g3, g4
   flaplacei.SetParameters(0, 1)
   g1 = flaplacei.DrawCopy() # TF1
   g1.SetLineColor(kRed)
   g1.SetLineWidth(1)

   flaplacei.SetParameters(0, 2)
   g2 = flaplacei.DrawCopy("same") #TF1
   g2.SetLineColor(kGreen)
   g2.SetLineWidth(1)
 
   flaplacei.SetParameters(2, 1)
   g3 = flaplacei.DrawCopy("same") #TF1
   g3.SetLineColor(kBlue)
   g3.SetLineWidth(1)

   flaplacei.SetParameters(2, 2)
   g4 = flaplacei.DrawCopy("same") #TF1
   g4.SetLineColor(kMagenta)
   g4.SetLineWidth(1)
   
   global legend2
   legend2 = TLegend(.7,.15,0.9,.35)
   legend2.AddEntry(f1,"alpha=0 beta=1","l")
   legend2.AddEntry(f2,"alpha=0 beta=2","l")
   legend2.AddEntry(f3,"alpha=2 beta=1","l")
   legend2.AddEntry(f4,"alpha=2 beta=2","l")
   legend2.Draw()
   c1.cd()
   


if __name__ == "__main__":
   mathLaplace()
