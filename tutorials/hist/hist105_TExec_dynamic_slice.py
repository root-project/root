## \file
## \ingroup tutorial_hist
## \notebook
## Example of function called when a mouse event occurs in a pad.
## When moving the mouse in the canvas, a second canvas shows the
## projection along X of the bin corresponding to the Y position
## of the mouse. The resulting histogram is fitted with a gaussian.
## A "dynamic" line shows the current bin position in Y.
## This more elaborated example can be used as a starting point
## to develop more powerful interactive applications exploiting Cling
## as a development engine.
##
## Note that a class is used to hold on to the canvas that display
## the selected slice.
##
## \macro_image
## \macro_code
##
## \author Rene Brun, Johann Cohen-Tanugi, Wim Lavrijsen, Enric Tejedor, Sergey Linev

import sys
import ctypes

from ROOT import gRandom, gPad, gROOT
from ROOT import kTRUE, kRed
from ROOT import TCanvas, TH2, TH2F


class DynamicExec:

   def __init__( self ):
      self._cX   = None
      self._cY   = None
      self._old  = None

   def __call__( self ):

      h = gPad.GetSelected();
      if not h:
         return

      if not isinstance( h, TH2 ):
         return

      if not gPad.FeedbackMode( kTRUE ):
         return

      px = gPad.GetEventX()
      py = gPad.GetEventY()
      upx = gPad.AbsPixeltoX( px )
      upy = gPad.AbsPixeltoY( py )

      uxmin, uxmax = gPad.GetUxmin(), gPad.GetUxmax()
      uymin, uymax = gPad.GetUymin(), gPad.GetUymax()

      # erase old position and paint lines at current position
      if self._old != None:
         gPad.PaintLine( uxmin, self._old[1], uxmax, self._old[1] )
         gPad.PaintLine( self._old[0], uymin, self._old[0], uymax )
      gPad.PaintLine( uxmin, upy, uxmax, upy )
      gPad.PaintLine( upx, uymin, upx, uymax )

      self._old = upx, upy

      x = gPad.PadtoX( upx )
      y = gPad.PadtoY( upy )

      padsav = gPad

    # create or set the display canvases
      if not self._cX:
         self._cX = TCanvas( 'c2', 'Projection Canvas in X', 730, 10, 700, 500 )
      else:
         self._DestroyPrimitive( 'X' )

      if not self._cY:
         self._cY = TCanvas( 'c3', 'Projection Canvas in Y', 10, 550, 700, 500 )
      else:
         self._DestroyPrimitive( 'Y' )

      self.DrawSlice( h, y, 'Y' )
      self.DrawSlice( h, x, 'X' )

      padsav.cd()

   def _DestroyPrimitive( self, xy ):
      proj = getattr( self, '_c'+xy ).GetPrimitive( 'Projection '+xy )
      if proj:
         proj.IsA().Destructor( proj )

   def DrawSlice( self, histo, value, xy ):
      yx = xy == 'X' and 'Y' or 'X'

    # draw slice corresponding to mouse position
      canvas = getattr( self, '_c'+xy )
      canvas.SetGrid()
      canvas.cd()

      bin = getattr( histo, 'Get%saxis' % xy )().FindBin( value )
      hp = getattr( histo, 'Projection' + yx )( '', bin, bin )
      hp.SetFillColor( 38 )
      hp.SetName( 'Projection ' + xy )
      hp.SetTitle( xy + 'Projection of bin=%d' % bin )
      hp.Fit( 'gaus', 'ql' )
      hp.GetFunction( 'gaus' ).SetLineColor( kRed )
      hp.GetFunction( 'gaus' ).SetLineWidth( 6 )
      canvas.Update()


if __name__ == '__main__':
 # create a new canvas.

   c1 = TCanvas('c1', 'Dynamic Slice Example', 10, 10, 700, 500 )
   c1.SetFillColor( 42 )
   c1.SetFrameFillColor( 33 )

 # create a 2-d histogram, fill and draw it
   hpxpy  = TH2F( 'hpxpy', 'py vs px', 40, -4, 4, 40, -4, 4 )
   hpxpy.SetStats( 0 )
   x, y = ctypes.c_double( 0.1 ), ctypes.c_double( 0.101 )
   for i in range( 50000 ):
   # pass ctypes doubles by reference, then retrieve their modified values with .value
     gRandom.Rannor( x, y )
     hpxpy.Fill( x.value, y.value )
   hpxpy.Draw( 'COL' )

 # Add a TExec object to the canvas (explicit use of __main__ is for IPython)
   import __main__
   __main__.slicer = DynamicExec()
   c1.AddExec( 'dynamic', 'TPython::Exec( "slicer()" );' )
   c1.Update()
