
# ruby-root testsuite
# port of the original $ROOT/hsum.C tutorial
# (20/01/2004)  --elathan  <elathan@phys.uoa.gr>
#
# original header:
# To see the output of this macro, click begin_html <a href="gif/hsum.gif" >here</a> end_html
# Simple example illustrating how to use the C++ interpreter	
# to fill histograms in a loop and show the graphics results

gROOT.Reset

c1 = TCanvas.new("c1","The HSUM example",200,10,600,400)
    c1.SetGrid

gBenchmark = TBenchmark.new.Start("hsum")

# Create some histograms.
total  = TH1F.new("total","This is the total distribution",100,-4,4)
main   = TH1F.new("main","Main contributor",100,-4,4)
s1     = TH1F.new("s1","This is the first signal",100,-4,4)
s2     = TH1F.new("s2","This is the second signal",100,-4,4)

    total.Sumw2   # this makes sure that the sum of squares of weights will be stored
    total.SetMarkerStyle(21)
    total.SetMarkerSize(0.7)
    main.SetFillColor(16)
    s1.SetFillColor(42)
    s2.SetFillColor(46)
    slider = nil 

# Fill histograms randomly

rnd = TRandom.new.SetSeed
kUPDATE = 500
10000.times do |i| 
    xmain = rnd.Gaus(-1,1.5)
    xs1   = rnd.Gaus(-0.5,0.5)
    xs2   = rnd.Landau(1,0.15)
    main.Fill(xmain)
    s1.Fill(xs1,0.3)
    s2.Fill(xs2,0.2)
    total.Fill(xmain)
    total.Fill(xs1,0.3)
    total.Fill(xs2,0.2)
    if (i && (i%kUPDATE) == 0) 
        if (i == kUPDATE) 
            total.Draw("e1p")
            main.Draw("same")
            s1.Draw("same")
            s2.Draw("same")
            c1.Update
            slider = TSlider.new("slider","test",4.2,0,4.6,total.GetMaximum,38)
            slider.SetFillColor(46)
        end
        slider.SetRange(0,i/10000.0) if slider
        c1.Modified
        c1.Update
     end
end

slider.SetRange(0,1)
total.Draw("sameaxis") # to redraw axis hidden by the fill area
c1.Modified
gBenchmark.Show("hsum")
gApplication.Run
