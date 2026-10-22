## \file
## \ingroup tutorial_math
## \notebook
##
## Test of the TMath.BetaDist and TMath.BetaDistI functions of ROOT.
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

#globals
gPad = ROOT.gPad


# void
def mathBeta() :
   
   global c1
   c1 = TCanvas("c1", "TMath::BetaDist",600,800)
   c1.Divide(1, 2)
   
   global pad1
   pad1 = c1.cd(1)
   pad1.SetGrid()
   
   # Defining TMath::BetaDist function.
   global fbeta
   fbeta = TF1("fbeta", "TMath::BetaDist(x, [0], [1])", 0, 1)
   
   # Defining and setting-up.
   global f1, f2, f3, f4 

   fbeta.SetParameters(0.5, 0.5)
   f1 = fbeta.DrawCopy()
   f1.SetLineColor(kRed)
   f1.SetLineWidth(1)

   fbeta.SetParameters(0.5, 2)
   f2 = fbeta.DrawCopy("same")
   f2.SetLineColor(kGreen)
   f2.SetLineWidth(1)
 
   fbeta.SetParameters(2, 0.5)
   f3 = fbeta.DrawCopy("same")
   f3.SetLineColor(kBlue)
   f3.SetLineWidth(1)
 
   fbeta.SetParameters(2, 2)
   f4 = fbeta.DrawCopy("same")
   f4.SetLineColor(kMagenta)
   f4.SetLineWidth(1)
 
   # Adding the first Legends.
   global legend1
   legend1 = TLegend(.5,.7,.8,.9)
   legend1.AddEntry(f1,"p=0.5  q=0.5","l")
   legend1.AddEntry(f2,"p=0.5  q=2","l")
   legend1.AddEntry(f3,"p=2    q=0.5","l")
   legend1.AddEntry(f4,"p=2    q=2","l")
   legend1.Draw()
   
   # Setting grid on second pad.
   global pad2
   pad2 = c1.cd(2)
   pad2.SetGrid()

   # Defining TMath::BetaDistI function.
   global fbetai
   fbetai = TF1("fbetai", "TMath::BetaDistI(x, [0], [1])", 0, 1)

   
   #Setting-up.
   global g1, g2, g3, g4
   fbetai.SetParameters(0.5, 0.5)
   g1 = fbetai.DrawCopy()
   g1.SetLineColor(kRed)
   g1.SetLineWidth(1)

   fbetai.SetParameters(0.5, 2)
   g2 = fbetai.DrawCopy("same")
   g2.SetLineColor(kGreen)
   g2.SetLineWidth(1)

   fbetai.SetParameters(2, 0.5)
   g3 = fbetai.DrawCopy("same")
   g3.SetLineColor(kBlue)
   g3.SetLineWidth(1)

   fbetai.SetParameters(2, 2)
   g4 = fbetai.DrawCopy("same")
   g4.SetLineColor(kMagenta)
   g4.SetLineWidth(1)
   
   # Adding the second legends.
   global legend2
   legend2 = TLegend(.7,.15,0.9,.35)
   legend2.AddEntry(f1,"p=0.5  q=0.5","l")
   legend2.AddEntry(f2,"p=0.5  q=2","l")
   legend2.AddEntry(f3,"p=2    q=0.5","l")
   legend2.AddEntry(f4,"p=2    q=2","l")
   legend2.Draw()
   c1.cd()
   


if __name__ == "__main__":
   mathBeta()
