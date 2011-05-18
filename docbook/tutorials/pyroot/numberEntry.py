from ROOT import *

class pMyMainFrame( TGMainFrame ):
   def __init__( self, parent, width, height ):
      TGMainFrame.__init__( self, parent, width, height )

      self.fHor1 = TGHorizontalFrame( self, 60, 20, kFixedWidth )
      self.fExit = TGTextButton( self.fHor1, "&Exit", "gApplication->Terminate(0)" )
      self.fExit.SetCommand( 'TPython::Exec( "raise SystemExit" )' )
      self.fHor1.AddFrame( self.fExit, TGLayoutHints( kLHintsTop | kLHintsLeft | 
                                                      kLHintsExpandX, 4, 4, 4, 4 ) )
      self.AddFrame( self.fHor1, TGLayoutHints( kLHintsBottom | kLHintsRight, 2, 2, 5, 1 ) )
   
      self.fNumber = TGNumberEntry( self, 0, 9,999, TGNumberFormat.kNESInteger,
                                               TGNumberFormat.kNEANonNegative, 
                                               TGNumberFormat.kNELLimitMinMax,
                                               0, 99999 )
      self.fLabelDispatch = TPyDispatcher( self.DoSetlabel )
      self.fNumber.Connect(
         "ValueSet(Long_t)", "TPyDispatcher", self.fLabelDispatch, "Dispatch()" )
      self.fNumber.GetNumberEntry().Connect(
         "ReturnPressed()", "TPyDispatcher", self.fLabelDispatch, "Dispatch()" )
      self.AddFrame( self.fNumber, TGLayoutHints( kLHintsTop | kLHintsLeft, 5, 5, 5, 5 ) )
      self.fGframe = TGGroupFrame( self, "Value" )
      self.fLabel = TGLabel( self.fGframe, "No input." )
      self.fGframe.AddFrame( self.fLabel, TGLayoutHints( kLHintsTop | kLHintsLeft, 5, 5, 5, 5) )
      self.AddFrame( self.fGframe, TGLayoutHints( kLHintsExpandX, 2, 2, 1, 1 ) )
   
      self.SetCleanup( kDeepCleanup )
      self.SetWindowName( "Number Entry" )
      self.MapSubwindows()
      self.Resize( self.GetDefaultSize() )
      self.MapWindow()

   def __del__( self ):
      self.Cleanup()

   def DoSetlabel( self ):
      self.fLabel.SetText( Form( "%d" % self.fNumber.GetNumberEntry().GetIntNumber() ) )
      self.fGframe.Layout()


if __name__ == '__main__':
   window = pMyMainFrame( gClient.GetRoot(), 50, 50 )

