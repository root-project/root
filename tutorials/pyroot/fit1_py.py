
c1 = TCanvas( 'c1', 'The Fit Canvas' )
c1.SetGridx()
c1.SetGridy()

fill = TFile( 'fillrandom.root' )
fill.ls()
sqroot.Print()

h1f.Fit( 'sqroot' )

fitlabel = TPaveText( 0.6, 0.3, 0.9, 0.80, 'NDC' )
fitlabel.SetTextAlign( 12 )
fitlabel.SetFillColor( 42 )
fitlabel.ReadFile( 'fit1_py.py' )
fitlabel.Draw()

