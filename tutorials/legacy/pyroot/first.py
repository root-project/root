## \file
## \ingroup tutorial_pyroot_legacy
## \notebook
## My first PyROOT interactive session
##
## \macro_image
## \macro_code
##
## \author Wim Lavrijsen

from ROOT import TCanvas, TF1, TPaveLabel, TPad, TText
from ROOT import gROOT


nut = TCanvas( 'nut', 'FirstSession', 100, 10, 700, 900 )
nut.Range( 0, 0, 20, 24 )
nut.SetFillColor( 10 )
nut.SetBorderSize( 2 )

pl = TPaveLabel( 3, 22, 17, 23.7, 'My first PyROOT interactive session', 'br' )
pl.SetFillColor( 18 )
pl.Draw()

t = TText( 0, 0, 'a' )
t.SetTextFont( 62 )
t.SetTextSize( 0.025 )
t.SetTextAlign( 12 )
t.DrawText( 2, 20.3, 'PyROOT provides ROOT bindings for Python, a powerful interpreter.' )
t.DrawText( 2, 19.3, 'Blocks of lines can be entered typographically.' )
t.DrawText( 2, 18.3, 'Previous typed lines can be recalled.' )

t.SetTextFont( 72 )
t.SetTextSize( 0.026 )
t.DrawText( 3, 17, r'>>>  x, y = 5, 7' )
t.DrawText( 3, 16, r'>>>  import math; x*math.sqrt(y)' )
t.DrawText( 3, 14, r'>>>  for i in range(2,7): print "sqrt(%d) = %f" % (i,math.sqrt(i))' )
t.DrawText( 3, 10, r'>>>  import ROOT; f1 = ROOT.TF1( "f1", "sin(x)/x", 0, 10 )' )
t.DrawText( 3,  9, r'>>>  f1.Draw()' )
t.SetTextFont( 81 )
t.SetTextSize( 0.018 )
t.DrawText( 4, 15,   '13.228756555322953' )
t.DrawText( 4, 13.3, 'sqrt(2) = 1.414214' )
t.DrawText( 4, 12.7, 'sqrt(3) = 1.732051' )
t.DrawText( 4, 12.1, 'sqrt(4) = 2.000000' )
t.DrawText( 4, 11.5, 'sqrt(5) = 2.236068' )
t.DrawText( 4, 10.9, 'sqrt(6) = 2.449490' )

pad = TPad( 'pad', 'pad', .2, .05, .8, .35 )
pad.SetFillColor( 42 )
pad.SetFrameFillColor( 33 )
pad.SetBorderSize( 10 )
pad.Draw()
pad.cd()
pad.SetGrid()

f1 = TF1( 'f1', 'sin(x)/x', 0, 10 )
f1.Draw()
nut.cd()
nut.Update()
