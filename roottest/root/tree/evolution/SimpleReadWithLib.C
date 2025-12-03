int SimpleReadWithLib(int mode = 0) {
   switch (mode) {
   case 0:
      break;
   case 1:
      gROOT->ProcessLine(".L SimpleOne.C+");
      break;
   case 2:
      gROOT->ProcessLine(".L SimpleTwo.C+");
      break;
   }
   TChain *c = new TChain("tree");
   c->Add("SimpleOne.root");
   c->Add("SimpleTwo.root");

   for(Long64_t i = 0; i < c->GetEntries(); ++i) {
      if (mode > 0) {
         void *p = TClass::GetClass("Simple")->New();
         c->SetBranchAddress("simpleSplit.",&p);
      }
      c->GetEntry(i);
   }
}
