## \file
## \ingroup tutorial_math
## \notebook
##
## Example on how to use the CrystalBall function and its distributions (pdf and cdf)
## pdf is short for Probability Density Function.
## cdf is short for Cumulative Density Function.
##
##
## \macro_image
## \macro_code
##
## \author Lorenzo Moneta
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
def CrystalBall() :
   
   global c1
   c1 = TCanvas()
   c1.Divide(1, 3)
   
   # crystal ball function
   c1.cd(1)
   
   global f1
   f1 = TF1("f1", "crystalball", -5, 5)
   f1.SetParameters(1, 0, 1, 2, 0.5)
   f1.SetLineColor(kRed)
   f1.Draw()

   # At using directly the function ROOT.Math note that its parameters' definition
   # are different. 
   # The order of parameters goes like: x, alpha, n, sigma, mean.
   # Use `help(ROOT.Math.crystalball_function)` to see its definition.
   """
   ROOT.Math.crystalball_function( x,  alpha,  n,  sigma,  mean = 0)
   """
   
   global f2
   f2 = TF1("f2", "ROOT::Math::crystalball_function(x, 2, 1, 1, 0)", -5, 5)
   f2.SetLineColor(kGreen)
   f2.Draw("same")
   
   global f3
   f3 = TF1("f3", "ROOT::Math::crystalball_function(x, 2, 2, 1, 0)", -5, 5)
   f3.SetLineColor(kBlue)
   f3.Draw("same")
   
   global legend
   legend = TLegend(0.7, 0.6, 0.9, 1.)
   legend.AddEntry(f1, "N=0.5 alpha=2", "L")
   legend.AddEntry(f2, "N=1   alpha=2", "L")
   legend.AddEntry(f3, "N=2   alpha=2", "L")
   legend.Draw()
   

   c1.cd(2)

   global pdf1, pdf2, pdf3 
   pdf1 = TF1("pdf", "crystalballn", -5, 5)
   pdf1.SetParameters(2, 0, 1, 2, 3)
   pdf1.Draw()

   pdf2 = TF1("pdf", "ROOT::Math::crystalball_pdf(x, 3, 1.01, 1, 0)", -5, 5)
   pdf2.SetLineColor(kBlue)
   pdf2.Draw("same")

   pdf3 = TF1("pdf", "ROOT::Math::crystalball_pdf(x, 2, 2, 1, 0)", -5, 5)
   pdf3.SetLineColor(kGreen)
   pdf3.Draw("same")
   
   global legend2
   legend2 = TLegend(0.7, 0.6, 0.9, 1.)
   legend2.AddEntry(pdf1, "N=3    alpha=2", "L")
   legend2.AddEntry(pdf2, "N=1.01 alpha=3", "L")
   legend2.AddEntry(pdf3, "N=2    alpha=3", "L")
   legend2.Draw()
   
   c1.cd(3)
   global cdf, cdfc
   cdf = TF1("cdf", "ROOT::Math::crystalball_cdf(x, 1.2, 2, 1, 0)", -5, 5)
   cdfc = TF1("cdfc", "ROOT::Math::crystalball_cdf_c(x, 1.2, 2, 1, 0)", -5, 5)
   
   #Setting-up 
   cdf.SetLineColor(kRed - 3)
   cdf.SetMinimum(0.)
   cdf.SetMaximum(1.)
   cdf.Draw()
   cdfc.SetLineColor(kMagenta)
   cdfc.Draw("Same")
   
   global legend3
   legend3 = TLegend(0.7, 0.7, 0.9, 1.)
   legend3.AddEntry(cdf, "N=1.2 alpha=2", "L")
   legend3.AddEntry(cdfc, "N=1.2 alpha=2", "L")
   legend3.Draw()
   


if __name__ == "__main__":
   CrystalBall()
