
# ruby-root testsuite
# port of the original $ROOT/hksimple.C tutorial
# (18/01/2004)  --elathan  <elathan@phys.uoa.gr>
#
# original header:
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#*-*
#*-*  This script illustrates the advantages of a TH1K histogram
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

def padRefresh(pad, flag=nil)
    return if not pad
    pad.Modified
    pad.Update
    pad.GetListOfPrimitives.each do |to|
        padRefresh(to.as("TPad"), 1) if to.InheritsFrom("TPad")
    end
    return if flag
    gSystem.ProcessEvents
end

#tapp = TApplication.new("rr: hksimple.rb")

# Create a new canvas.
c1 = TCanvas.new("c1","Dynamic Filling Example",200,10,600,900)
    c1.SetFillColor(42)

# Create a normal histogram and two TH1K histograms
hpx = []
    hpx << TH1F.new("hp0","Normal histogram",1000,-4,4)
    hpx << TH1K.new("hk1","Nearest Neighboor of order 3",1000,-4,4)
    hpx << TH1K.new("hk2","Nearest Neighboor of order 16",1000,-4,4,16)
    
    c1.Divide(1,3)
    
    hpx.each_with_index do |h, i|
        c1.cd i+1
        gPad.SetFrameFillColor(33)
        h.SetFillColor 48
        h.Draw 
    end

    gRandom.SetSeed
    $kUPDATE = 10
    300.times do |i| 
        px = gRandom.Gaus
        py = gRandom.Gaus
        hpx.each do |h|
            h.Fill(px)
        end
        padRefresh(c1.as("TPad")) if (i && (i % $kUPDATE) == 0)
    end

    hpx.each do |h|
        h.Fit("gaus", "", "")
    end
  
    padRefresh(c1)
gApplication.Run
