{
   TChain *c = new TChain("tree");
   c->Add("SimpleOne.root");
   c->Add("SimpleTwo.root");
   Long64_t n = c->Scan("simple.fData");
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(n==0);
#else
   return (n==0);
#endif
}
