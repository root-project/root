## \file
## \ingroup tutorial_pyroot
## Tornado example.
## \notebook
##
## \macro_image
## \macro_code
##
## \author Wim Lavrijsen

from ROOT import TCanvas, TView, TPolyMarker3D, TPaveText
from ROOT import gROOT, gBenchmark
from math import cos, sin, pi

gBenchmark.Start( 'tornado' )

d = 16
numberOfPoints = 200
numberOfCircles = 40

# create and open a canvas
sky = TCanvas( 'sky', 'Tornado', 300, 10, 700, 500 )
sky.SetFillColor( 14 )

# creating view
view = TView.CreateView()
rng = numberOfCircles * d
view.SetRange( 0, 0, 0, 4.0*rng, 2.0*rng, rng )

polymarkers = []
for j in range( d, numberOfCircles * d, d ):

 # create a PolyMarker3D
   pm3d = TPolyMarker3D( numberOfPoints )

 # set points
   for i in range( 1, numberOfPoints ) :
      csin = sin( 2*pi / numberOfPoints * i ) + 1
      ccos = cos( 2*pi / numberOfPoints  * i ) + 1
      esin = sin( 2*pi / (numberOfCircles*d) * j ) + 1
      x = j * ( csin + esin );
      y = j * ccos;
      z = j;
      pm3d.SetPoint( i, x, y, z );

 # set marker size, color & style
   pm3d.SetMarkerSize( 1 )
   pm3d.SetMarkerColor( 2 + ( d == ( j & d ) ) )
   pm3d.SetMarkerStyle( 3 )

 # draw
   pm3d.Draw()

 # save a reference
   polymarkers.append( pm3d )

gBenchmark.Show( 'tornado' )

ct = gBenchmark.GetCpuTime( 'tornado' )
timeStr = 'Execution time: %g sec.' % ct

text = TPaveText( 0.1, 0.81, 0.9, 0.97 )
text.SetFillColor( 42 )
text.AddText( 'PyROOT example: tornado.py' )
text.AddText( timeStr )
text.Draw()

sky.Update()
