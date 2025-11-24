## \file
## \ingroup tutorial_graphics
## \notebook
## Example illustrating how to draw a TGaxis-object with 
## labels defined by a function.
##
## \macro_image
## \macro_code
##
## \author  Olivier Couet
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TF1 = ROOT.TF1
TH1F = ROOT.TH1F
TH2F = ROOT.TH2F
TGaxis = ROOT.TGaxis

#globals
gStyle = ROOT.gStyle




# void
def gaxis2():

   #Setting-up formal style.
   gStyle.SetOptStat(0)
   

   global h2
   h2 = TH2F("h","Axes",100,0,10,100,-2,2)
   h2.Draw()
   

   global f1
   f1 = TF1("f1","-x",-10,10)

   global A1
   A1 = TGaxis(0,2,10,2,"f1",510,"-")
   A1.SetTitle("axis with decreasing values")
   A1.Draw()
   

   global f2
   f2 = TF1("f2","exp(x)",0,2)

   global A2
   A2 = TGaxis(1,1,9,1,"f2")
   A2.SetTitle("exponential axis")
   A2.SetLabelSize(0.03)
   A2.SetTitleSize(0.03)
   A2.SetTitleOffset(1.2)
   A2.Draw()
   

   global f3
   f3 = TF1("f3","log10(x)",1,1000)

   global A3
   A3 = TGaxis(2,-2,2,0,"f3",505,"G")
   A3.SetTitle("logarithmic axis")
   A3.SetLabelSize(0.03)
   A3.SetTitleSize(0.03)
   A3.SetTitleOffset(1.2)
   A3.Draw()
   


if __name__ == "__main__":
   gaxis2()
