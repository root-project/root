## \file
## \ingroup tutorial_graphics
## \notebook -js
## This script displays interpreted functions with TFormula and TF1.
##
## \macro_image
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT

#classes
TCanvas = ROOT.TCanvas
TFormula = ROOT.TFormula
TF1 = ROOT.TF1

#globals
gObjectTable = ROOT.gObjectTable



# void
def formula1() :

   # Defining and setting-up the canvas.
   global c1
   c1 = TCanvas("c1","Example with Formula",200,10,700,500)
   c1.SetGridx()
   c1.SetGridy()

   #
   # We create a formula object and compute the value of this formula
   # for two different values of the x variable.
   #
   global form1
   form1 = TFormula("form1","sqrt(abs(x))")
   form1.Eval(2)
   form1.Eval(-45)

   #
   # Create a one dimensional function and draw it
   #
   global fun1
   fun1 = TF1("fun1","abs(sin(x)/x)",0,10)
   fun1.Draw()

   # Upding the canvas so that our function is draw on it. Only fun1 is draw-on.
   c1.Update()

   #
   # Before leaving this demo, we print the list of objects known to ROOT
   #
   if (gObjectTable) : gObjectTable.Print()
   


if __name__ == "__main__":
   formula1()
