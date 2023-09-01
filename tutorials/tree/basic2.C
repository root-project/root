/// \file
/// \ingroup tutorial_tree
/// \notebook -js
/// Create can ntuple reading data from an ascii file.
/// This macro is a variant of basic.C
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void basic2() {
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/tree/");
   dir.ReplaceAll("/./","/");

   TFile *f = new TFile("basic2.root","RECREATE");
   TH1F *h1 = new TH1F("h1","x distribution",100,-4,4);
   TTree *T = new TTree("ntuple","data from ascii file");
   Long64_t nlines = T->ReadFile(Form("%sbasic.dat",dir.Data()),"x:y:z");
   printf(" found %lld points\n",nlines);
   T->Draw("x","z>2");
   T->Write();
}
