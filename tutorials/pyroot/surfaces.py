## \file
## \ingroup tutorial_pyroot
## \notebook
## Surfaces example
##
## \macro_image
## \macro_code
##
## \author Wim Lavrijsen

from ROOT import TCanvas, TPaveText, TPad, TF2
from ROOT import gROOT, gStyle



c1 = TCanvas( 'c1', 'Surfaces Drawing Options', 200, 10, 700, 900 )
c1.SetFillColor( 42 )
gStyle.SetFrameFillColor( 42 )
title = TPaveText( .2, 0.96, .8, .995 )
title.SetFillColor( 33 )
title.AddText( 'Examples of Surface options' )
title.Draw()

pad1 = TPad( 'pad1', 'Gouraud shading', 0.03, 0.50, 0.98, 0.95, 21 )
pad2 = TPad( 'pad2', 'Color mesh',      0.03, 0.02, 0.98, 0.48, 21 )
pad1.Draw()
pad2.Draw()

# We generate a 2-D function
f2 = TF2( 'f2', 'x**2 + y**2 - x**3 -8*x*y**4', -1, 1.2, -1.5, 1.5 )
f2.SetContour( 48 )
f2.SetFillColor( 45 )

# Draw this function in pad1 with Gouraud shading option
pad1.cd()
pad1.SetPhi( -80 )
pad1.SetLogz()
f2.Draw( 'surf4' )

# Draw this function in pad2 with color mesh option
pad2.cd()
pad2.SetTheta( 25 )
pad2.SetPhi( -110 )
pad2.SetLogz()
f2.Draw( 'surf1' )

c1.Update()
