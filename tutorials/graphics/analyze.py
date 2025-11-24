## \file
## \ingroup tutorial_graphics
## \notebook -js
##
## This script produces a flowchart of the TFormula.Analyze.
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
TPaveText = ROOT.TPaveText
TText = ROOT.TText
TLine = ROOT.TLine
TArrow = ROOT.TArrow


# void
def analyze() :

   global c1
   c1 = TCanvas("c1", "Analyze.mac", 620, 790)
   c1.Range(-1, 0, 19, 30)

   global pl1
   pl1 = TPaveLabel(0, 27, 3.5, 29, "Analyze")
   pl1.SetFillColor(42)
   pl1.Draw()

   global pt1
   pt1 = TPaveText(0, 22.8, 4, 25.2)
   t1 = pt1.AddText("Parenthesis matching")
   t2 = pt1.AddText("Remove unnecessary")
   t2a = pt1.AddText("parenthesis")
   pt1.Draw()

   global pt2
   pt2 = TPaveText(6, 23, 10, 25)
   t3 = pt2.AddText("break of")
   t4 = pt2.AddText("Analyze")
   pt2.Draw()

   global pt3
   pt3 = TPaveText(0, 19, 4, 21)
   t4 = pt3.AddText("look for simple")
   t5 = pt3.AddText("operators")
   pt3.Draw()

   global pt4
   pt4 = TPaveText(0, 15, 4, 17)
   t6 = pt4.AddText("look for an already")
   t7 = pt4.AddText("defined expression")
   pt4.Draw()

   global pt5
   pt5 = TPaveText(0, 11, 4, 13)
   t8 = pt5.AddText("look for usual")
   t9 = pt5.AddText("functions :cos sin ..")
   pt5.Draw()

   global pt6
   pt6 = TPaveText(0, 7, 4, 9)
   t10 = pt6.AddText("look for a")
   t11 = pt6.AddText("numeric value")
   pt6.Draw()

   global pt7
   pt7 = TPaveText(6, 18.5, 10, 21.5)
   t12 = pt7.AddText("Analyze left and")
   t13 = pt7.AddText("right part of")
   t14 = pt7.AddText("the expression")
   pt7.Draw()

   global pt8
   pt8 = TPaveText(6, 15, 10, 17)
   t15 = pt8.AddText("Replace expression")
   pt8.Draw()

   global pt9
   pt9 = TPaveText(6, 11, 10, 13)
   t16 = pt9.AddText("Analyze")
   pt9.SetFillColor(42)
   pt9.Draw()

   global pt10
   pt10 = TPaveText(6, 7, 10, 9)
   t17 = pt10.AddText("Error")
   t18 = pt10.AddText("Break of Analyze")
   pt10.Draw()

   global pt11
   pt11 = TPaveText(14, 22, 17, 24)
   pt11.SetFillColor(42)
   t19 = pt11.AddText("Analyze")
   t19a = pt11.AddText("Left")
   pt11.Draw()

   global pt12
   pt12 = TPaveText(14, 19, 17, 21)
   pt12.SetFillColor(42)
   t20 = pt12.AddText("Analyze")
   t20a = pt12.AddText("Right")
   pt12.Draw()

   global pt13
   pt13 = TPaveText(14, 15, 18, 18)
   t21 = pt13.AddText("StackNumber++")
   t22 = pt13.AddText("operator[StackNumber]")
   t23 = pt13.AddText("= operator found")
   pt13.Draw()

   global pt14
   pt14 = TPaveText(12, 10.8, 17, 13.2)
   t24 = pt14.AddText("StackNumber++")
   t25 = pt14.AddText("operator[StackNumber]")
   t26 = pt14.AddText("= function found")
   pt14.Draw()

   global pt15
   pt15 = TPaveText(6, 7, 10, 9)
   t27 = pt15.AddText("Error")
   t28 = pt15.AddText("break of Analyze")
   pt15.Draw()

   global pt16
   pt16 = TPaveText(0, 2, 7, 5)
   t29 = pt16.AddText("StackNumber++")
   t30 = pt16.AddText("operator[StackNumber] = 0")
   t31 = pt16.AddText("value[StackNumber] = value found")
   pt16.Draw()

   global ar
   ar = TArrow(2, 27, 2, 25.4, 0.012, "|>")
   ar.SetFillColor(1)
   ar.Draw()
   ar.DrawArrow(2, 22.8, 2, 21.2, 0.012, "|>")
   ar.DrawArrow(2, 19, 2, 17.2, 0.012, "|>")
   ar.DrawArrow(2, 15, 2, 13.2, 0.012, "|>")
   ar.DrawArrow(2, 11, 2, 9.2, 0.012, "|>")
   ar.DrawArrow(2, 7, 2, 5.2, 0.012, "|>")
   ar.DrawArrow(4, 24, 6, 24, 0.012, "|>")
   ar.DrawArrow(4, 20, 6, 20, 0.012, "|>")
   ar.DrawArrow(4, 16, 6, 16, 0.012, "|>")
   ar.DrawArrow(4, 12, 6, 12, 0.012, "|>")
   ar.DrawArrow(4, 8, 6, 8, 0.012, "|>")
   ar.DrawArrow(10, 20, 14, 20, 0.012, "|>")
   ar.DrawArrow(12, 23, 14, 23, 0.012, "|>")
   ar.DrawArrow(12, 16.5, 14, 16.5, 0.012, "|>")
   ar.DrawArrow(10, 12, 12, 12, 0.012, "|>")

   global ta
   ta = TText(2.2, 22.2, "err = 0")
   ta.SetTextFont(71)
   ta.SetTextSize(0.015)
   ta.SetTextColor(4)
   ta.SetTextAlign(12)
   ta.Draw()
   ta.DrawText(2.2, 18.2, "not found")
   ta.DrawText(2.2, 6.2, "found")

   global tb
   tb = TText(4.2, 24.1, "err != 0")
   tb.SetTextFont(71)
   tb.SetTextSize(0.015)
   tb.SetTextColor(4)
   tb.SetTextAlign(11)
   tb.Draw()
   tb.DrawText(4.2, 20.1, "found")
   tb.DrawText(4.2, 16.1, "found")
   tb.DrawText(4.2, 12.1, "found")
   tb.DrawText(4.2, 8.1, "not found")

   global l1
   l1 = TLine(12, 16.5, 12, 23)
   l1.Draw()
   


if __name__ == "__main__":
   analyze()
