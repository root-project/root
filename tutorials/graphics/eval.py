## \file
## \ingroup tutorial_graphics
## \notebook -js
##
## This script produces the flowchart of TFormula.Eval.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TLine = ROOT.TLine
TPaveLabel = ROOT.TPaveLabel
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow



# void
def eval() :
   
   global c1
   c1 = TCanvas("c1")
   c1.Range(0,0,20,10)

   global pt1
   pt1 = TPaveLabel(0.2,4,3,6,"Eval")
   pt1.SetTextSize(0.5)
   pt1.SetFillColor(42)
   pt1.Draw()

   global pt2
   pt2 = TPaveText(4.5,4,7.8,6)
   pt2.Draw()
   t1 = pt2.AddText("Read Operator")
   t2 = pt2.AddText("number i")

   global pt3, t4, t5, t6
   pt3 = TPaveText(9,3.5,17.5,6.5)
   t4 = pt3.AddText("Apply Operator to current stack values")
   t5 = pt3.AddText("Example: if operator +")
   t6 = pt3.AddText("value[i] += value[i-1]")
   t4.SetTextAlign(22)
   t5.SetTextAlign(22)
   t6.SetTextAlign(22)
   t5.SetTextColor(4)
   t6.SetTextColor(2)
   pt3.Draw()

   global pt4
   pt4 = TPaveLabel(4,0.5,12,2.5,"return result = value[i]")
   pt4.Draw()

   global ar1
   ar1 = TArrow(6,4,6,2.7,0.02,"|>")
   ar1.Draw()

   global t7
   t7 = TText(6.56,2.7,"if i = number of stack elements")
   t7.SetTextSize(0.04)
   t7.Draw()
   ar1.DrawArrow(6,8,6,6.2,0.02,"|>")

   global l1
   l1 = TLine(12,6.6,12,8)
   l1.Draw()
   l1.DrawLine(12,8,6,8)
   ar1.DrawArrow(3,5,4.4,5,0.02,"|>")
   ar1.DrawArrow(7.8,5,8.9,5,0.02,"|>")
   



if __name__ == "__main__":
   eval()
