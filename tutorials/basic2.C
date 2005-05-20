{
//   example of macro to create can ntuple reading data from an ascii file.
//   This macro is a variant of basic.C
   
   gROOT->Reset();
   TFile *f = new TFile("basic2.root","RECREATE");
   TH1F *h1 = new TH1F("h1","x distribution",100,-4,4);
   TTree *T = new TTree("ntuple","data from ascii file");
   Long64_t nlines = T->ReadFile("basic.dat","x:y:z");
   printf(" found %lld points\n",nlines);
   T->Draw("x","z>2");
   T->Write();
}
