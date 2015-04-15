import os
from ROOT import TCanvas, TPaveText
from ROOT import gROOT



chelp = TCanvas( 'chelp', 'Help to run demos', 200, 10, 700, 500 )

welcome = TPaveText( .1, .8, .9, .97 )
welcome.AddText( 'Welcome to the ROOT demos' )
welcome.SetTextFont( 32 )
welcome.SetTextColor( 4 )
welcome.SetFillColor( 24 )
welcome.Draw()

hdemo = TPaveText( .05, .05, .95, .7 )
hdemo.SetTextAlign( 12 )
hdemo.SetTextFont( 52 )

text = """- Run demo hsimple.py first. Then in any order
- Click left mouse button to execute one demo
- Click right mouse button to see the title of the demo
- Click on 'Close Bar' to exit from the demo menu
- Select 'File/Print' to print a Postscript view of the canvas
- You can execute a demo with the mouse or type commands
- During the demo (try on this canvas) you can:
  .... Use left button to move/grow/etc objects
  .... Use middle button to pop overlapping objects
  .... Use right button to get an object sensitive pop-up
 """
for line in text.split( os.linesep ):
   hdemo.AddText( line )

hdemo.SetAllWith( '....', 'color', 2 )
hdemo.SetAllWith( '....', 'font', 72 )
hdemo.SetAllWith( '....', 'size', 0.04 )

hdemo.Draw()
chelp.Update()
