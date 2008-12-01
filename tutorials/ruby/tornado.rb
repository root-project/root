#
# Ruby version of tornado.C. 
#
# This example tests static method calls (TView.CreateView()).
#
gROOT.Reset()

gBenchmark = TBenchmark.new().Start('tornado')

d = 16
numberOfPoints = 200
numberOfCircles = 40

# create and open a canvas
sky = TCanvas.new('sky', 'Tornado', 300, 10, 700, 500)
sky.SetFillColor(14)

# creating view
view = TView.CreateView(1, 0, 0)
rng = numberOfCircles * d
view.SetRange(0, 0, 0, 4.0*rng, 2.0*rng, rng)

polymarkers = []

d.step(numberOfCircles * d, d) do |j|	
	# create a PolyMarker3D
	pm3d = TPolyMarker3D.new( numberOfPoints )
 	# set points
	for  i in 1..numberOfPoints  do
		csin = Math.sin( 2*Math::PI / numberOfPoints * i ) + 1
      	ccos = Math.cos( 2*Math::PI / numberOfPoints  * i ) + 1
      	esin = Math.sin( 2*Math::PI / (numberOfCircles*d) * j ) + 1
      	x = j * ( csin + esin );
      	y = j * ccos;
      	z = j;
      	pm3d.SetPoint( i, x, y, z );
	end
 	# set marker size, color & style
   	pm3d.SetMarkerSize( 1 )
   	pm3d.SetMarkerColor( 2 + (( d == ( j & d ) ) ? 1 : 0))
   	pm3d.SetMarkerStyle( 3 )

 	# draw
   	pm3d.Draw()
 	# save a reference
   	polymarkers << pm3d 
end

gBenchmark.Show( 'tornado' )

ct = gBenchmark.GetCpuTime( 'tornado' )
timeStr = 'Execution time: %g sec.' % ct

text = TPaveText.new( 0.1, 0.81, 0.9, 0.97 )
text.SetFillColor( 42 )
text.AddText( 'Ruby ROOT example: tornado.rb' )
text.AddText( timeStr )
text.Draw()

sky.Update()
gApplication.Run()
