{
   TChain *c = new TChain("tree");
   c->Add("SimpleOne.root");
   c->Add("SimpleTwo.root");
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   Long64_t n ; n = c->Scan("simple.fData");
#else
   Long64_t n = c->Scan("simple.fData");
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(n==0);
#else
   return (n==0);
#endif
}
