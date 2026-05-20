void abcwrite(const char* mode)
{
   Holder h;

   TFile f(TString::Format("abc_%s.root", mode), "recreate");
   TTree* tree = new TTree("tree", "abc tree");
   tree->Branch("h", &h);
   for (int e = 0; e < 100; ++e) {
      h.Set(e);
	   tree->Fill();
   }
   tree->Write();
   delete tree;
}
