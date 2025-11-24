## \file
## \ingroup tutorial_graphics
## \notebook -js
##
## This macro produces the flowchart of TFormula:Compile
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TArrow = ROOT.TArrow
TPaveText = ROOT.TPaveText




# void
def compile() :

   global c1
   c1 = TCanvas("c1")
   c1.Range(0,0,1,1)

   global ptc
   ptc = TPaveLabel(0.02,0.42,0.2,0.58,"Compile")
   ptc.SetTextSize(0.40)
   ptc.SetFillColor(32)
   ptc.Draw()

   global psub
   psub = TPaveText(0.28,0.4,0.65,0.6)
   psub.Draw()
   t2 = psub.AddText("Substitute some operators")
   t3 = psub.AddText("to C++ style")

   global panal
   panal = TPaveLabel(0.73,0.42,0.98,0.58,"Analyze")
   panal.SetTextSize(0.40)
   panal.SetFillColor(42)
   panal.Draw()

   global ar1
   ar1 = TArrow(0.2,0.5,0.27,0.5,0.02,"|>")
   ar1.SetLineWidth(6)
   ar1.SetLineColor(4)
   ar1.Draw()

   global ar2
   ar2 = TArrow(0.65,0.5,0.72,0.5,0.02,"|>")
   ar2.SetLineWidth(6)
   ar2.SetLineColor(4)
   ar2.Draw()
   


if __name__ == "__main__":
   compile()
