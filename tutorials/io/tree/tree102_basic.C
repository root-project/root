/// \file
/// \ingroup tutorial_tree
/// \notebook -js
/// Create can ntuple reading data from an ascii file.
/// This macro is a variant of tree101_basic.C
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void tree102_basic()
{
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/io/tree/");
   dir.ReplaceAll("/./", "/");

   auto f = TFile::Open("tree102.root", "RECREATE");
   auto h1 = new TH1F("h1", "x distribution", 100, -4, 4);
   auto T = new TTree("ntuple", "data from ascii file");
   Long64_t nlines = T->ReadFile(TString::Format("%sbasic.dat", dir.Data()), "x:y:z");
   printf(" found %lld points\n", nlines);
   T->Draw("x", "z>2");
   T->Write();
}
