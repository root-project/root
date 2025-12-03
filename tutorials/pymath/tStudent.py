## \file
## \ingroup tutorial_math
## \notebook
##
## Example-script describing the very well known t-Student-distribution.
##
## ~~~{.cpp}
## IP[0]: %run tStudent.py
## ~~~
##
## From the t-Student distribution,
## this script draws its Probability Density Function(pdf) and 
## its Cumulative Density Function(cdf). Then draws 
## 10 quantiles of it. 
##
## \macro_image
## \macro_code
##
## \author Magdalena Slawinska
## \translator P. P.


import ROOT

TH1D = ROOT.TH1D
TF1 = ROOT.TF1 
TCanvas = ROOT.TCanvas 
TSystem = ROOT.TSystem 
TLegend = ROOT.TLegend 
TLegendEntry = ROOT.TLegendEntry 
TString = ROOT.TString

#math
Math = ROOT.Math
#DistFunc = Math.DistFunc 

#types
Double_t = ROOT.Double_t
Int_t = ROOT.Int_t

#constants
kRed = ROOT.kRed
kWhite = ROOT.kWhite
kBlue = ROOT.kBlue

#globals
gPad = ROOT.gPad





# void
def tStudent() :
   
   # In case you have an old version of ROOT and you'll
   # need to load manually the library libMathMore.so.
   #ROOT.gSystem.Load("libMathMore");'''
   
   # This is the way to force load of MathMore.
   ROOT.Math.MathMoreLibrary.Load()
   
   n = Int_t(100)
   a = Double_t( -5. )
   b = Double_t( 5. )
   #r  = Double_t(3)
   global pdf, cum
   pdf = TF1("pdf", "ROOT::Math::tdistribution_pdf(x,3.0)", a,b)
   cum = TF1("cum", "ROOT::Math::tdistribution_cdf(x,3.0)", a,b)
   
   global quant
   quant = TH1D("quant", "", 9, 0, 0.9)
   
   #for(int i=1; i < 10; i++)
   for i in range(1, 10, 1):
      quant.Fill((i-0.5)/10.0, ROOT.Math.tdistribution_quantile((1.0*i)/10, 3.0 ) )
   
   # Setting-up xx, a.k.a the quantiles.
   xx = [ Double_t() ] *10 
   xx[0] = -1.5
   #for(int i=1; i<9; i++)
   for i in range(1, 9, 1):
      xx[i] = quant.GetBinContent(i)
   xx[9] = 1.5

   # Initializing the histograms with the quantiles.
   global pdfq
   pdfq = [TH1D() for _ in range(10) ]
   #nbin = n//10.0
   #for(int i=0; i < 9; i++) {
   for i in range(0, 9, 1):

      nbin = n * (xx[i+1]-xx[i])/3.0 + 1.0
      nbin = int(nbin)
      name = TString("pdf")
      name += TString(str(i))
      pdfq[i] = TH1D(str(name), "", nbin, xx[i], xx[i+1] )
      
      #for(int j=1; j<nbin; j++) {
      j = 1
      while( j < nbin ):
         x = j * (xx[i+1]-xx[i])/nbin + xx[i]
         pdfq[i].SetBinContent(j, ROOT.Math.tdistribution_pdf(x,3))
         j += 1
         
      
   
   global Canvas
   Canvas = TCanvas("DistCanvas", "Student Distribution graphs", 10, 10, 800, 700)

   # Setting-up the elements of the Canvas
   pdf.SetTitle("Student t distribution function")
   cum.SetTitle("Cumulative for Student t")
   quant.SetTitle("10-quantiles  for Student t")

   Canvas.Divide(2, 2)

   # Drawing each element on each sub-canvas.
   Canvas.cd(1)
   pdf.SetLineWidth(2)
   pdf.DrawCopy()

   Canvas.cd(2)
   cum.SetLineWidth(2)
   cum.SetLineColor(kRed)
   cum.Draw()

   Canvas.cd(3)
   quant.Draw()
   quant.SetLineWidth(2)
   quant.SetLineColor(kBlue)
   quant.SetStats(False)

   Canvas.cd(4)
   # Note: 
   # Important:
   # We draw first the pdf to set-up fourth pad. Like Range, RangeAxis, ...
   pdf.Draw("") #  

   # Drawing the quantiles.
   #for(int i=0; i < 9; i++) {
   for i in range(0, 9, 1):
      pdfq[i].SetStats(False)
      pdfq[i].SetFillColor(i+1)
      pdfq[i].Draw("same")

   # Second, we set a new title for pdf.
   pdf.SetTitle("Student t & its quantiles")
   # and we draw on the same pad(the fourth).
   pdf.Draw("same")
      
   # We update our Canvas.
   # We update our Canvas.
   Canvas.Modified()
   Canvas.cd()
   Canvas.Draw()
   


if __name__ == "__main__":
   tStudent()
