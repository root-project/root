
gROOT.Reset()

c1 = TCanvas.new( "c1", "Help to run demos", 200, 10, 700, 500 )

welcome = TPaveText.new(0.1, 0.8, 0.9, 0.97 )
welcome.AddText( "Welcome to the ROOT demos" )
welcome.SetTextFont( 32 )
welcome.SetTextColor( 4 )
welcome.SetFillColor( 24 )
welcome.Draw()

hdemo = TPaveText.new( 0.05, 0.05, 0.95, 0.7 )
hdemo.SetTextAlign( 12 )
hdemo.SetTextFont( 52 )
hdemo.AddText( "- Run demo hsimple.rb first. Then in any order" )
hdemo.AddText( "- Click left mouse button to execute one demo" )
hdemo.AddText( "- Click right mouse button to see the title of the demo" )
hdemo.AddText( "- Click on 'Close Bar' to exit from the demo menu" )
hdemo.AddText( "- Select 'File/Print' to print a Postscript view of the canvas" )
hdemo.AddText( "- You can execute a demo with the mouse or type commands" )
hdemo.AddText( "- During the demo (try on this canvas) you can :" )
hdemo.AddText( ".... Use left button to move/grow/etc objects" )
hdemo.AddText( ".... Use middle button to pop overlapping objects" )
hdemo.AddText( ".... Use right button to get an object sensitive pop-up" )
hdemo.AddText( " " )
hdemo.SetAllWith( "....", "color", 2 )
hdemo.SetAllWith( "....", "font", 72 )
hdemo.SetAllWith( "....", "size", 0.04 )

hdemo.Draw()
c1.Update()
gApplication.Run