{
   TChain *c = new TChain("tree");
   c->Add("SimpleOne.root");
   c->Add("SimpleTwo.root");
   Long64_t n = c->Scan("simple.fData");
   return (n==0);
}
