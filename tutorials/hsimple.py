## \file
## \ingroup Tutorials
## \notebook -js
##  This program creates :
##    - a one dimensional histogram
##    - a two dimensional histogram
##    - a profile histogram
##    - a memory-resident ntuple
##
##  These objects are filled with some random numbers and saved on a file.
##
## \macro_image
## \macro_code
##
## \author Wim Lavrijsen, Enric Tejedor
## \author Vincenzo Eduardo Padulano (CERN), 09.2025

import numpy
from ROOT import TH1F, TH2F, TCanvas, TFile, TNtuple, TProfile, gBenchmark, gSystem

# Create a new canvas, and customize it.
c1 = TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500)
c1.SetFillColor(42)
frame = c1.GetFrame()
frame.SetFillColor(21)
frame.SetBorderSize(6)
frame.SetBorderMode(-1)

# Create some histograms, a profile histogram and an ntuple
hpx = TH1F("hpx", "This is the px distribution", 100, -4, 4)
hpxpy = TH2F("hpxpy", "py vs px", 40, -4, 4, 40, -4, 4)
hprof = TProfile("hprof", "Profile of pz versus px", 100, -4, 4, 0, 20)
ntuple = TNtuple("ntuple", "Demo ntuple", "px:py:pz:random:i")

# Set canvas/frame attributes.
hpx.SetFillColor(48)

gBenchmark.Start("hsimple")

# Fill histograms randomly.
for i in range(25000):
    # Retrieve the generated values
    px, py = numpy.random.standard_normal(size=2)

    pz = px * px + py * py
    random = numpy.random.rand()

    # Fill histograms.
    hpx.Fill(px)
    hpxpy.Fill(px, py)
    hprof.Fill(px, pz)
    ntuple.Fill(px, py, pz, random, i)

    # Update display every 1000 events.
    if i > 0 and i % 1000 == 0:
        hpx.Draw()

        c1.Modified()
        c1.Update()

        if gSystem.ProcessEvents():  # allow user interrupt
            break

gBenchmark.Show("hsimple")

# Create a new ROOT binary machine independent file.
# Note that this file may contain any kind of ROOT objects, histograms,
# pictures, graphics objects, detector geometries, tracks, events, etc.
with TFile("py-hsimple.root", "RECREATE", "Demo ROOT file with histograms") as hfile:
    # Save all created objects in the file
    hfile.WriteObject(hpx)
    hfile.WriteObject(hpxpy)
    hfile.WriteObject(hprof)
    # TNTuple is special because it is a TTree-derived class. To make sure all the
    # dataset is properly written to disk, we connect it to the file and then
    # we ask the ntuple to write all the information to the file itself.
    ntuple.SetDirectory(hfile)
    ntuple.Write()

c1.Modified()
c1.Update()
