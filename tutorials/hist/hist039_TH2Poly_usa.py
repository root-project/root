# \file
# \ingroup tutorial_hist
# \notebook -js
# \preview This tutorial illustrates how to create an histogram with polygonal
# bins (TH2Poly), fill it and draw it using the `col` option. The initial data
# are stored in TMultiGraphs. They represent the USA map. Such histograms can
# be rendered in 3D using the option `legogl`.
#
# The initial data have been downloaded from: http://www.maproom.psu.edu/dcw/
# This database was developed in 1991/1992 and national boundaries reflect
# political reality as of that time.
#
# \macro_code
# \macro_image
#
# \date February 2024
# \author Olivier Couet

import ROOT
import os.path
import urllib.request

states = ["alabama",      "arizona",      "arkansas",       "california",    "colorado",      "connecticut",   "delaware",
          "florida",      "georgia",      "idaho",          "illinois",      "indiana",       "iowa",          "kansas",
          "kentucky",     "louisiana",    "maine",          "maryland",      "massachusetts", "michigan",      "minnesota",
          "mississippi",  "missouri",     "montana",        "nebraska",      "nevada",        "new_hampshire", "new_jersey",
          "new_mexico",   "new_york",     "north_carolina", "north_dakota",  "ohio",          "oklahoma",      "oregon",
          "pennsylvania", "rhode_island", "south_carolina", "south_dakota",  "tennessee",     "texas",         "utah",
          "vermont",      "virginia",     "washington",     "west_virginia", "wisconsin",     "wyoming"]

pops = [4708708,  6595778, 2889450,  36961664, 5024748, 3518288,  885122,  18537969, 9829211, 1545801,
        12910409, 6423113, 3007856,  2818747,  4314113, 4492076,  1318301, 5699478,  6593587, 9969727,
        5266214,  2951996, 5987580,  974989,   1796619, 2643085,  1324575, 8707739,  2009671, 19541453,
        9380884,  646844,  11542645, 3687050,  3825657, 12604767, 1053209, 4561242,  812383,  6296254,
        24782302, 2784572, 621760,   7882590,  6664195, 1819777,  5654774, 544270]

usa = ROOT.TCanvas("USA", "USA")
usa.ToggleEventStatus()
lon1 = -130
lon2 = -65
lat1 = 24
lat2 = 50
p = ROOT.TH2Poly("Lower48", "Lower 48 Population (2009);Latitude;Longitude", lon1, lon2, lat1, lat2)

fileName = "usa.root"
fileNameUrl = "http://root.cern/files/usa.root"
if not os.path.isfile(fileName):
    urllib.request.urlretrieve(fileNameUrl, fileName)
f = ROOT.TFile.Open(fileName)

# Define the TH2Poly bins.
mg = ROOT.TMultiGraph
for key in f.GetListOfKeys():
    obj = key.ReadObj()
    if (obj.InheritsFrom("TMultiGraph")):
          p.AddBin(obj)

# Fill TH2Poly, with capital letters for the states names
for state, pop in zip (states, pops):
    p.Fill(state, pop)

ROOT.gStyle.SetOptStat(0)
p.Draw("colz textn")

# Add the reference for the population
pupulationRef = ROOT.TLatex(-128, 27, "#scale[.55]{#splitline{Source:}{http://eadiv.state.wy.us/pop/st-09est.htm}}")
pupulationRef.DrawClone()

