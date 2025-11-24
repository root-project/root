## \file
## \ingroup tutorial_graphics
## \notebook
##
## This script draws some Feynman diagrams.
##
## \macro_image
## \macro_code
##
## \author Otto Schaile
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TLine = ROOT.TLine
TCurlyArc = ROOT.TCurlyArc
TCurlyLine = ROOT.TCurlyLine
TArc = ROOT.TArc
TLatex = ROOT.TLatex

#maths
sqrt = ROOT.sqrt

#globals
gStyle = ROOT.gStyle



# void
def feynman() :

   global c1
   c1 = TCanvas("c1", "A canvas", 10,10, 600, 300)
   c1.Range(0, 0, 140, 60)
   
   global linsav
   linsav = gStyle.GetLineWidth()
   gStyle.SetLineWidth(3)
   
   global t
   t = TLatex()
   t.SetTextAlign(22)
   t.SetTextSize(0.1)

   global l1
   l1 = TLine(10, 10, 30, 30)
   l1.Draw()

   global l2
   l2 = TLine(10, 50, 30, 30)
   l2.Draw()

   global ginit
   ginit = TCurlyArc(30, 30, 12.5*sqrt(2), 135, 225)
   ginit.SetWavy()
   ginit.Draw()
   t.DrawLatex(7,6,"e^{-}")
   t.DrawLatex(7,55,"e^{+}")
   t.DrawLatex(7,30,"#gamma")
   

   global Gamma
   Gamma = TCurlyLine(30, 30, 55, 30)
   Gamma.SetWavy()
   Gamma.Draw()
   t.DrawLatex(42.5,37.7,"#gamma")
   

   global a
   a = TArc(70, 30, 15)
   a.Draw()
   t.DrawLatex(55, 45,"#bar{q}")
   t.DrawLatex(85, 15,"q")

   global gluon
   gluon = TCurlyLine(70, 45, 70, 15)
   gluon.Draw()
   t.DrawLatex(77.5,30,"g")
   

   global z0
   z0 = TCurlyLine(85, 30, 110, 30)
   z0.SetWavy()
   z0.Draw()
   t.DrawLatex(100, 37.5,"Z^{0}")
   

   global l3
   l3 = TLine(110, 30, 130, 10)
   l3.Draw()

   global l4
   l4 = TLine(110, 30, 130, 50)
   l4.Draw()
   

   global gluon1
   gluon1 = TCurlyArc(110, 30, 12.5*sqrt(2), 315, 45)
   gluon1.Draw()
   
   t.DrawLatex(135,6,"#bar{q}")
   t.DrawLatex(135,55,"q")
   t.DrawLatex(135,30,"g")

   c1.Update()

   gStyle.SetLineWidth(linsav)
   


if __name__ == "__main__":
   feynman()
