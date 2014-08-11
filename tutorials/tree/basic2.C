void basic2() {
//   example of macro to create can ntuple reading data from an ascii file.
//   This macro is a variant of basic.C
//Author: Rene Brun

   TString dir = gSystem->UnixPathName(__FILE__);
   dir.ReplaceAll("basic2.C","");
   dir.ReplaceAll("/./","/");

   TFile *f = new TFile("basic2.root","RECREATE");
   TH1F *h1 = new TH1F("h1","x distribution",100,-4,4);
   TTree *T = new TTree("ntuple","data from ascii file");
   Long64_t nlines = T->ReadFile(Form("%sbasic.dat",dir.Data()),"x:y:z");
   printf(" found %lld points\n",nlines);
   T->Draw("x","z>2");
   T->Write();
}
