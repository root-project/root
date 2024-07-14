## \file
## \ingroup tutorial_graphics
## \notebook
##
## This scripts draws a pave text.
## Text lines are added in to use the AddText method.
## Line separator can be added using the AddLine-method.
##
## AddText-method returns a TText-object corresponding to the line added 
## at doing the pave. This return value 
## can be used to modify text attributes.
##
## Once the TPaveText-object is built, the text of each line can be 
## retrieved as a TText-object with the GetLine and GetLineWith methods;
## wich are also useful to modify the text
## attributes of a line.
##
## Enjoy!
##
## \macro_image
## \macro_code
##
## \author Olivier Couet
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TPaveLabel = ROOT.TPaveLabel
TPaveText = ROOT.TPaveText

#constants
kOrange = ROOT.kOrange
kBlue = ROOT.kBlue



# TCanvas
def pavetext() :

   global c, pt
   c = TCanvas("c")
   pt = TPaveText(.05,.1,.95,.8)
   
   pt.AddText("A TPaveText can contain severals line of text.")
   pt.AddText("They are added to the pave using the AddText method.")
   pt.AddLine(.0,.5,1.,.5)
   pt.AddText("Even complex TLatex formulas can be added:")
   
   global t1
   t1 = pt.AddText("F(t) = #sum_{i=-#infty}^{#infty}A(i)cos#[]{#frac{i}{t+i}}")
   
   t1.SetTextColor(kBlue)
   
   pt.Draw()
   
   t2 = pt.GetLineWith("Even")
   t2.SetTextColor(kOrange+1)
   
   return c
   


if __name__ == "__main__":
   pavetext()
