#
# This macro generates
# a begin_html <a href="gif/na49canvas.gif">Canvas</a> end_html
# with 2 views of the NA49 detector.
#

import ROOT

ROOT.gROOT.Reset();
c1 = ROOT.TCanvas( 'c1', 'The NA49 canvas', 200, 10, 700, 780 )

ROOT.gBenchmark.Start( 'na49view' )

all = ROOT.TPad( 'all', 'A Global view of NA49', 0.02, 0.02, 0.48, 0.82, 28 )
tof = ROOT.TPad( 'tof', 'One Time Of Flight element', 0.52, 0.02, 0.98, 0.82, 28 )
all.Draw();
tof.Draw();
na49title = ROOT.TPaveLabel( 0.04, 0.86, 0.96, 0.98, 'Two views of the NA49 detector' )
na49title.SetFillColor( 32 )
na49title.Draw()
#
nageom = ROOT.TFile( 'na49.root' )
n49 = ROOT.gROOT.FindObject( 'na49' )
n49.SetBomb( 1.2 )
n49.cd()     # Set current geometry
all.cd()     # Set current pad
n49.Draw()
c1.Update()
tof.cd()
TOFR1 = n49.GetNode( 'TOFR1' )
TOFR1.Draw()
c1.Update()

ROOT.gBenchmark.Show( 'na49view' )

# To have a better and dynamic view of any of these pads,
# you can click with the middle button of your mouse to select it.
# Then select "View with x3d" in the VIEW menu of the Canvas.
# Once in x3d, you are in wireframe mode by default.
# You can switch to:
#   - Hidden Line mode by typing E
#   - Solid mode by typing R
#   - Wireframe mode by typing W
#   - Stereo mode by clicking S (and you need special glasses)
#   - To leave x3d type Q
