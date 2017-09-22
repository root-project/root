## \file
## \ingroup tutorial_tdataframe
## \notebook -draw
## This tutorial shows the possibility to use data models which are more
## complex than flat ntuples with TDataFrame
##
## \macro_code
##
## \date May 2017
## \author Danilo Piparo

import ROOT

# A simple helper function to fill a test tree: this makes the example stand-alone.
fill_tree_code = '''

using FourVector = ROOT::Math::XYZTVector;
using FourVectors = std::vector<FourVector>;
using CylFourVector = ROOT::Math::RhoEtaPhiVector;

void fill_tree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   FourVectors tracks;
   t.Branch("tracks", &tracks);

   const double M = 0.13957; // set pi+ mass
   TRandom3 R(1);

   for (int i = 0; i < 50; ++i) {
      auto nPart = R.Poisson(15);
      tracks.clear();
      tracks.reserve(nPart);
      for (int j = 0; j < nPart; ++j) {
         double px = R.Gaus(0, 10);
         double py = R.Gaus(0, 10);
         double pt = sqrt(px * px + py * py);
         double eta = R.Uniform(-3, 3);
         double phi = R.Uniform(0.0, 2 * TMath::Pi());
         CylFourVector vcyl(pt, eta, phi);
         // set energy
         double E = sqrt(vcyl.R() * vcyl.R() + M * M);
         FourVector q(vcyl.X(), vcyl.Y(), vcyl.Z(), E);
         // fill track vector
         tracks.emplace_back(q);
      }
      t.Fill();
   }

   t.Write();
   f.Close();
   return;
}
'''

# We prepare an input tree to run on
fileName = "tdf002_dataModel_py.root"
treeName = "myTree"
ROOT.gInterpreter.Declare(fill_tree_code)
ROOT.fill_tree(fileName, treeName)

# We read the tree from the file and create a TDataFrame, a class that
# allows us to interact with the data contained in the tree.
TDF = ROOT.ROOT.Experimental.TDataFrame
d = TDF(treeName, fileName)

# Operating on branches which are collection of objects
# Here we deal with the simplest of the cuts: we decide to accept the event
# only if the number of tracks is greater than 5.
n_cut = 'tracks.size() > 8'
nentries = d.Filter(n_cut).Count();

print("%s passed all filters" %nentries.GetValue())

# Another possibility consists in creating a new column containing the
# quantity we are interested in.
# In this example, we will cut on the number of tracks and plot their
# transverse momentum.

getPt_code ='''
std::vector<double> getPt (const FourVectors &tracks)
{
   std::vector<double> pts;
   pts.reserve(tracks.size());
   for (auto &t : tracks) pts.emplace_back(t.Pt());
   return pts;
}
'''
ROOT.gInterpreter.Declare(getPt_code)

getPtWeights_code ='''
std::vector<double> getPtWeights (const FourVectors &tracks) {
   std::vector<double> ptsw;
   ptsw.reserve(tracks.size());
   for (auto &t : tracks) ptsw.emplace_back(1. / t.Pt());
   return ptsw;
};
'''
ROOT.gInterpreter.Declare(getPtWeights_code)

augmented_d = d.Define('tracks_n', '(int)tracks.size()') \
               .Filter('tracks_n > 2') \
               .Define('tracks_pts', 'getPt( tracks )') \
               .Define("tracks_pts_weights", 'getPtWeights( tracks )' )

# The histogram is initialised with a tuple containing the parameters of the
# histogram
trN = augmented_d.Histo1D(("", "", 40, -.5, 39.5), "tracks_n")
trPts = augmented_d.Histo1D("tracks_pts")
trWPts = augmented_d.Histo1D("tracks_pts", "tracks_pts_weights")

c1 = ROOT.TCanvas()
trN.Draw()
c1.Print("tracks_n.png")

c2 = ROOT.TCanvas()
trPts.Draw()
c2.Print("tracks_pt.png")

c3 = ROOT.TCanvas()
trWPts.Draw()
c3.Print("tracks_Wpt.png")
