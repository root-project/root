/// \file
/// \ingroup tutorial_tdataframe
/// \notebook -nodraw
/// This tutorial shows how to fill any object the class of which exposes a
/// `Fill` method.
/// \macro_code
///
/// \date March 2017
/// \author Danilo Piparo


// A simple helper function to fill a test tree: this makes the example
// stand-alone.
void fill_tree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   double b1;
   float b2;
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   for (int i = 0; i < 100; ++i) {
      b1 = i;
      b2 = i * i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

int tdf005_fillAnyObject()
{

   // We prepare an input tree to run on
   auto fileName = "tdf005_fillAnyObject.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   // We read the tree from the file and create a TDataFrame.
   ROOT::Experimental::TDataFrame d(treeName, fileName);

   // ## Filling any object
   // We now fill some objects which are instances of classes which expose a
   // `Fill` method with some input arguments.
   auto th1d = d.Fill<double>(TH1D("th1d", "th1d", 64, 0, 128), {"b1"});
   auto th1i = d.Fill<float>(TH1I("th1i", "th1i", 64, 0, 128), {"b2"});
   auto th2d = d.Fill<double, float>(TH2D("th2d", "th2d", 64, 0, 128, 64, 0, 1024), {"b1", "b2"});

   auto c1 = new TCanvas();
   th1d->DrawClone();

   auto c2 = new TCanvas();
   th1i->DrawClone();

   auto c3 = new TCanvas();
   th2d->DrawClone("COLZ");

   return 0;
}
