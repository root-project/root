## \file
## \ingroup tutorial_pyroot
## \notebook -js
## TF1 example.
##
## \macro_image
## \macro_code
##
## \author Wim Lavrijsen

import ROOT

c1 = ROOT.TCanvas( 'c1', 'Example with Formula', 200, 10, 700, 500 )

# We create a formula object and compute the value of this formula
# for two different values of the x variable.
form1 = ROOT.TFormula( 'form1', 'sqrt(abs(x))' )
ROOT.SetOwnership(form1, False)
form1.Eval( 2 )
form1.Eval( -45 )

# Create a one dimensional function and draw it
fun1 = ROOT.TF1( 'fun1', 'abs(sin(x)/x)', 0, 10 )
ROOT.SetOwnership(fun1, False)
c1.SetGridx()
c1.SetGridy()
fun1.Draw()
c1.Update()

# Before leaving this demo, we print the list of objects known to ROOT
#
if ( ROOT.gObjectTable ):
    ROOT.gObjectTable.Print()
