#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
#*-*
#*-*  This program creates :
#*-*    - a one dimensional histogram
#*-*    - a two dimensional histogram
#*-*    - a profile histogram
#*-*    - a memory-resident ntuple
#*-*
#*-*  These objects are filled with some random numbers and saved on a file.
#*-*
#*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

gROOT.Reset

# Create a new canvas.
c1 = TCanvas.new('c1', 'Dynamic Filling Example', 200, 10, 700, 500)
c1.SetFillColor(42)
c1.GetFrame.SetFillColor(21)
c1.GetFrame.SetBorderSize(6)
c1.GetFrame.SetBorderMode(-1)

# Create a new ROOT binary machine independent file.
# Note that this file may contain any kind of ROOT objects, histograms,
# pictures, graphics objects, detector geometries, tracks, events, etc..
# This file is now becoming the current directory.

hfile = gROOT.FindObject('hsimple.root')
hfile.Close if hfile

hfile = TFile.new( 'hsimple.root', 'RECREATE', 'Demo ROOT file with histograms' )

# Create some histograms, a profile histogram and an ntuple
hpx    = TH1F.new('hpx', 'This is the px distribution', 100, -4, 4)
hpxpy  = TH2F.new('hpxpy', 'py vs px', 40, -4, 4, 40, -4, 4 )
hprof  = TProfile.new('hprof', 'Profile of pz versus px', 100, -4, 4, 0, 20)
ntuple = TNtuple.new('ntuple', 'Demo ntuple', 'px:py:pz:random:i')

#  Set canvas/frame attributes (save old attributes)
hpx.SetFillColor(48) 
gBenchmark = TBenchmark.new.Start('hsimple')

# Fill histograms randomly
rnd = TRandom.new.SetSeed
kUPDATE = 1000
25000.times do |i| 
   px = rnd.Gaus
   py = rnd.Gaus
   pz = px*px + py*py
   random = rnd.Rndm(1)
   hpx.Fill(px)
   hpxpy.Fill(px, py)
   hprof.Fill(px, pz)
   ntuple.Fill(px, py, pz, random, i)
   if i and i%kUPDATE == 0 
      hpx.Draw if i == kUPDATE
      c1.Modified
      c1.Update
      next if gSystem.ProcessEvents 
   end
end

gBenchmark.Show('hsimple')

# Save all objects in this file
hpx.SetFillColor(0)
hfile.Write
hpx.SetFillColor(48)
c1.Modified
c1.Update
gApplication.Run  

