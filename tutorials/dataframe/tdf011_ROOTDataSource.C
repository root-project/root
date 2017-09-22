/// \file
/// \ingroup tutorial_tdataframe
/// \notebook
/// This tutorial illustrates how use the TDataFrame in combination with a
/// TDataSource. In this case we use a TRootDS. This data source allows to read
/// a ROOT dataset from a TDataFrame in a different way, not based on the
/// regular TDataFrame code. This allows to perform all sorts of consistency
/// checks and illustrate the usage of the TDataSource in a didactic context.
///
/// \macro_code
///
/// \date September 2017
/// \author Danilo Piparo

void fill_tree(const char *fileName, const char *treeName)
{
   TFile f(fileName, "RECREATE");
   TTree t(treeName, treeName);
   int b1;
   t.Branch("b1", &b1);
   for (int i = 0; i < 10000; ++i) {
      b1 = i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

using TDS = ROOT::Experimental::TDF::TDataSource;
using TRootDS = ROOT::Experimental::TDF::TRootDS;

int tdf011_ROOTDataSource()
{
   auto fileName = "tdf011_ROOTDataSources.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   auto d_s = ROOT::Experimental::TDF::MakeRootDataFrame(treeName, fileName);

   /// Now we have a regular TDataFrame: the ingestion of data is delegated to
   /// the TDataSource. At this point everything works as before.
   auto h_s = d_s.Define("x", "1./(b1 + 1.)").Histo1D({"h_s", "h_s", 128, 0, .6}, "x");

   /// Now we redo the same with a TDF and we draw the two histograms
   ROOT::Experimental::TDataFrame d(treeName, fileName);
   auto h = d.Define("x", "1./(b1 + 1.)").Histo1D({"h", "h", 128, 0, .6}, "x");

   auto c_s = new TCanvas();
   c_s->SetLogy();
   h_s->DrawClone();

   auto c = new TCanvas();
   c->SetLogy();
   h->DrawClone();

   return 0;
}
