{
   TFile *file = TFile::Open("badvector.root");
   if (!file || file->IsZombie()) {
      Error("assertSparceSelection","Can not open badvector.root");
      return 1;
   }
   TTree *t = file->Get<TTree>("t");
   if (!t) {
      Error("assertSparceSelection","Can not find TTree name 't' in badvector.root");
      return 2;
   }
   Long64_t res = t->Draw("v","v<0");
   if (res != 14) {
      Error("assertSparceSelection","Wrong number of selected values: %lld instead of 14", res);
      return 3;
   }
   TH1F *h = htemp;
   if (h->GetEntries() != 14) {
      Error("assertSparceSelection","Wrong number of entries in histo: %lld instead of 14", res);
      return 3;
   }
   if (h->GetMean() < -0.305 || h->GetMean() > -0.304) {
      Error("assertSparceSelection","Histo contains unexpected values");
      h->Print("all");
      return 4;
   }
   return 0;
}
