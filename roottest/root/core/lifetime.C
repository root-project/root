{
   std::unique_ptr<TH1D> h1(new TH1D("h1", "h", 1, 0., 1.)); // attached to gROOT
   new TMemFile("tmp.root", "RECREATE");
   std::unique_ptr<TH1D> h2(new TH1D("h2", "h", 1, 0., 1.)); // attached to the intentionally not deleted TFile 
   return 0;
}
