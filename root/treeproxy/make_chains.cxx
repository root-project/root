#include "TChain.h"

TChain* make_long_chain()
{
   TChain *lchain = new TChain("longtree");
   lchain->Add("longtree1.root");
   lchain->Add("longtree2.root");
   return lchain;
}

TChain* make_float_chain()
{
   TChain *fchain = new TChain("floattree");
   fchain->Add("floattree1.root");
   fchain->Add("floattree2.root");
   return fchain;
}

