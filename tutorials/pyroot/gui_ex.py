## \file
## \ingroup tutorial_pyroot
## A Simple GUI Example
##
## \macro_code
##
## \author Wim Lavrijsen
from __future__ import print_function

import os, sys, ROOT

def pygaus( x, par ):
   import math
   if (par[2] != 0.0):
      arg1 = (x[0]-par[1])/par[2]
      arg2 = (0.01*0.39894228)/par[2]
      arg3 = par[0]/(1+par[3])

      gauss = arg3*arg2*math.exp(-0.5*arg1*arg1)
   else:
      print('returning 0')
      gauss = 0.
   return gauss

tpygaus = ROOT.TF1( 'pygaus', pygaus, -4, 4, 4 )
tpygaus.SetParameters( 1., 0., 1. )

def MyDraw():
   btn = ROOT.BindObject( ROOT.gTQSender, ROOT.TGTextButton )
   if btn.WidgetId() == 10:
      global tpygaus, window
      tpygaus.Draw()
      ROOT.gPad.Update()

m = ROOT.TPyDispatcher( MyDraw )


class pMainFrame( ROOT.TGMainFrame ):
   def __init__( self, parent, width, height ):
       ROOT.TGMainFrame.__init__( self, parent, width, height )

       self.Canvas    = ROOT.TRootEmbeddedCanvas( 'Canvas', self, 200, 200 )
       self.AddFrame( self.Canvas, ROOT.TGLayoutHints() )
       self.ButtonsFrame = ROOT.TGHorizontalFrame( self, 200, 40 )

       self.DrawButton   = ROOT.TGTextButton( self.ButtonsFrame, '&Draw', 10 )
       self.DrawButton.Connect( 'Clicked()', "TPyDispatcher", m, 'Dispatch()' )
       self.ButtonsFrame.AddFrame( self.DrawButton, ROOT.TGLayoutHints() )

       self.ExitButton   = ROOT.TGTextButton( self.ButtonsFrame, '&Exit', 20 )
       self.ExitButton.SetCommand( 'TPython::Exec( "raise SystemExit" )' )
       self.ButtonsFrame.AddFrame( self.ExitButton, ROOT.TGLayoutHints() )

       self.AddFrame( self.ButtonsFrame, ROOT.TGLayoutHints() )

       self.SetWindowName( 'My first GUI' )
       self.MapSubwindows()
       self.Resize( self.GetDefaultSize() )
       self.MapWindow()

   def __del__(self):
       self.Cleanup()


if __name__ == '__main__':
   window = pMainFrame( ROOT.gClient.GetRoot(), 200, 200 )
