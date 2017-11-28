/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -nodraw
/// This tutorial illustrates how to use TProfiles in combination with the
/// TDataFrame. See the documentation of TProfile and TProfile2D to better
/// understand the analogy of this code with the example one.
///
/// \macro_code
///
/// \date February 2017
/// \author Danilo Piparo

// A simple helper function to fill a test tree: this makes the example
// stand-alone.
void fill_tree(const char *treeName, const char *fileName)
{
   ROOT::Experimental::TDataFrame d(25000);
   d.Define("px", []() { return gRandom->Gaus(); })
      .Define("py", []() { return gRandom->Gaus(); })
      .Define("pz", [](double px, double py) { return sqrt(px * px + py * py); }, {"px", "py"})
      .Snapshot(treeName, fileName);
}

void tdf003_profiles()
{
   // We prepare an input tree to run on
   auto fileName = "tdf003_profiles.root";
   auto treeName = "myTree";
   fill_tree(treeName, fileName);

   // We read the tree from the file and create a TDataFrame.
   ROOT::Experimental::TDataFrame d(treeName, fileName, {"px", "py", "pz"});

   // Create the profiles
   auto hprof1d = d.Profile1D({"hprof1d", "Profile of pz versus px", 64, -4, 4});
   auto hprof2d = d.Profile2D({"hprof2d", "Profile of pz versus px and py", 40, -4, 4, 40, -4, 4, 0, 20});

   // And Draw
   auto c1 = new TCanvas("c1", "Profile histogram example", 200, 10, 700, 500);
   hprof1d->DrawClone();
   auto c2 = new TCanvas("c2", "Profile2D histogram example", 200, 10, 700, 500);
   c2->cd();
   hprof2d->DrawClone();
}
