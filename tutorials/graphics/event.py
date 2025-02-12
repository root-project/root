## \file
## \ingroup tutorial_graphics
## \notebook -js
##
## This script illustrates some basic primitives of TEvent-class.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.



import ROOT

#classes
TLine = ROOT.TLine

TCanvas = ROOT.TCanvas


TPaveText = ROOT.TPaveText
TText = ROOT.TText
TArrow = ROOT.TArrow



# void
def event() :

   global c1
   c1 = TCanvas("c1","ROOT Event description",700,500)
   c1.Range(0,0,14,15.5)

   global event
   event = TPaveText(1,13,3,15)
   event.SetFillColor(11)
   event.Draw()
   event.AddText("Event")

   global line
   line = TLine(1.1,13,1.1,1.5)
   line.SetLineWidth(2)
   line.Draw()
   line.DrawLine(1.3,13,1.3,3.5)
   line.DrawLine(1.5,13,1.5,5.5)
   line.DrawLine(1.7,13,1.7,7.5)
   line.DrawLine(1.9,13,1.9,9.5)
   line.DrawLine(2.1,13,2.1,11.5)

   global arrow
   arrow = TArrow(1.1,1.5,3.9,1.5,0.02,"|>")
   arrow.SetFillStyle(1001)
   arrow.SetFillColor(1)
   arrow.Draw()
   arrow.DrawArrow(1.3,3.5,3.9,3.5,0.02,"|>")
   arrow.DrawArrow(1.5,5.5,3.9,5.5,0.02,"|>")
   arrow.DrawArrow(1.7,7.5,3.9,7.5,0.02,"|>")
   arrow.DrawArrow(1.9,9.5,3.9,9.5,0.02,"|>")
   arrow.DrawArrow(2.1,11.5,3.9,11.5,0.02,"|>")

   global p1
   p1 = TPaveText(4,1,11,2)
   p1.SetTextAlign(12)
   p1.SetFillColor(42)
   p1.AddText("1 Mbyte")
   p1.Draw()

   global p2
   p2 = TPaveText(4,3,10,4)
   p2.SetTextAlign(12)
   p2.SetFillColor(42)
   p2.AddText("100 Kbytes")
   p2.Draw()

   global p3
   p3 = TPaveText(4,5,9,6)
   p3.SetTextAlign(12)
   p3.SetFillColor(42)
   p3.AddText("10 Kbytes")
   p3.Draw()

   global p4
   p4 = TPaveText(4,7,8,8)
   p4.SetTextAlign(12)
   p4.SetFillColor(42)
   p4.AddText("1 Kbytes")
   p4.Draw()

   global p5
   p5 = TPaveText(4,9,7,10)
   p5.SetTextAlign(12)
   p5.SetFillColor(42)
   p5.AddText("100 bytes")
   p5.Draw()

   global p6
   p6 = TPaveText(4,11,6,12)
   p6.SetTextAlign(12)
   p6.SetFillColor(42)
   p6.AddText("10 bytes")
   p6.Draw()
   
   global text
   text = TText()
   text.SetTextAlign(12)
   text.SetTextSize(0.04)
   text.SetTextFont(72)
   text.DrawText(6.2,11.5,"Header:Event_flag")
   text.DrawText(7.2,9.5,"Trigger_Info")
   text.DrawText(8.2,7.5,"Muon_Detector: TOF")
   text.DrawText(9.2,5.5,"Calorimeters")
   text.DrawText(10.2,3.5,"Forward_Detectors")
   text.DrawText(11.2,1.5,"TPCs")
   


if __name__ == "__main__":
   event()
