{
   TChain *c = new TChain("tree");
   c->Add("SimpleOne.root");
   c->Add("SimpleTwo.root");
   return !c->Scan();
}