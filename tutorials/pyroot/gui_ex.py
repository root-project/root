import os, sys, ROOT

def MyDraw():
   ROOT.gROOT.ProcessLine( 'TGButton* btn = (TGButton*)gTQSender;' )
   print 'MyDraw', ROOT.btn.WidgetId()

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
