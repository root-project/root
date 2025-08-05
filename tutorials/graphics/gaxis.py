## \file
## \ingroup tutorial_graphics
## \notebook
##
## A simple example illustrating how to draw TGaxis objects in various formats.
##
## \macro_image
## \macro_code
##
## \authors Rene Brun, Olivier Couet
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TGaxis = ROOT.TGaxis


# void
def gaxis() :

   global c1
   c1 = TCanvas("c1","Examples of TGaxis",10,10,700,500)
   c1.Range(-10,-1,10,1)
   

   global axis1
   axis1 = TGaxis(-4.5,-0.2,5.5,-0.2,-6,8,510,"")
   axis1.Draw()
   

   global axis2
   axis2 = TGaxis(-4.5,0.2,5.5,0.2,0.001,10000,510,"G")
   axis2.Draw()
   

   global axis3
   axis3 = TGaxis(-9,-0.8,-9,0.8,-8,8,50510,"")
   axis3.SetTitle("axis3")
   axis3.SetTitleOffset(0.5)
   axis3.Draw()
   

   global axis4
   axis4 = TGaxis(-7,-0.8,-7,0.8,1,10000,50510,"G")
   axis4.SetTitle("axis4")
   axis4.Draw()
   

   global axis5
   axis5 = TGaxis(-4.5,-0.6,5.5,-0.6,1.2,1.32,80506,"-+")
   axis5.SetLabelSize(0.03)
   axis5.SetTextFont(72)
   axis5.Draw()
   

   global axis6
   axis6 = TGaxis(-4.5,0.5,5.5,0.5,100,900,50510,"-")
   axis6.Draw()
   

   global axis7
   axis7 = TGaxis(-5.5,0.85,5.5,0.85,0,4.3e-6,510,"")
   axis7.Draw()
   

   global axis8
   axis8 = TGaxis(8,-0.8,8,0.8,0,9000,50510,"+L")
   axis8.Draw()
   
   # One can make a vertical axis going top->bottom. However the two x values should be
   # slightly different to avoid labels overlapping.

   global axis9
   axis9 = TGaxis(6.5,0.8,6.499,-0.8,0,90,50510,"-")
   axis9.Draw()
   


if __name__ == "__main__":
   gaxis()
