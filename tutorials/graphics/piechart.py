## \file
## \ingroup tutorial_graphics
## \notebook
##
## A simple Pie chart example.
##
## \macro_image
## \macro_code
##
## \authors Olivier Couet, Guido Volpi
## \translator P. P.


import ROOT
import ctypes

#classes
TCanvas = ROOT.TCanvas
TPie = ROOT.TPie

#types
c_double = ctypes.c_double
c_int = ctypes.c_int

#utils
def to_c_double( ls ):
   return (c_double * len(ls) )( * ls )
def to_c_int( ls ):
   return (c_int * len(ls) )( * ls )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# void
def piechart() :

   #data
   vals = [ .2 , 1.1 , .6 , .9 , 2.3 ]
   colors = [ 2 , 3 , 4 , 5 , 6 ]
   #to C-types
   vals = to_c_double( vals )
   colors = to_c_int( colors )
   
   nvals = len(vals)
   
   global cpie
   cpie = TCanvas("cpie","TPie test",700,700)
   cpie.Divide(2,2)
  
   global pie1, pie2, pie3, pie4
   pie1 = TPie("pie1",
      "Pie with offset and no colors",nvals,vals)
   pie2 = TPie("pie2",
      "Pie with radial labels",nvals,vals,colors)
   pie3 = TPie("pie3",
      "Pie with tangential labels",nvals,vals,colors)
   pie4 = TPie("pie4",
      "Pie with verbose labels",nvals,vals,colors)
   
   cpie.cd(1)
   pie1.SetAngularOffset(30.)
   pie1.SetEntryRadiusOffset( 4, 0.1)
   pie1.SetRadius(.35)
   pie1.Draw("3d")
   
   cpie.cd(2)
   pie2.SetEntryRadiusOffset(2,.05)
   pie2.SetEntryLineColor(2,2)
   pie2.SetEntryLineWidth(2,5)
   pie2.SetEntryLineStyle(2,2)
   pie2.SetEntryFillStyle(1,3030)
   pie2.SetCircle(.5,.45,.3)
   pie2.Draw("rsc")
   
   cpie.cd(3)
   pie3.SetY(.32)
   pie3.GetSlice(0).SetValue(.8)
   pie3.GetSlice(1).SetFillStyle(3031)
   pie3.SetLabelsOffset(-.1)
   pie3.Draw("3d t nol")
   pieleg = pie3.MakeLegend()
   pieleg.SetY1(.56)
   pieleg.SetY2(.86)
   
   cpie.cd(4)
   pie4.SetRadius(.2)
   pie4.SetLabelsOffset(.01)
   pie4.SetLabelFormat("#splitline{%val (%perc)}{%txt}")
   pie4.Draw("nol <")
   


if __name__ == "__main__":
   piechart()
