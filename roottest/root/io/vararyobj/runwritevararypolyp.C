// writevararypolyp.C

#include "vararypolyp.C"

#include "TFile.h"
#include "TTree.h"

#include <iostream>

using namespace std;

const char* testfilename = "vararypolyp.root";

void runwritevararypolyp()
{
  C* c = new C();
  c->set();
  TFile* f = new TFile(testfilename, "recreate");
  c->Write();
  TTree* t = new TTree("t", "t");
  t->Branch("br1.", &c);
  t->Fill();
  t->Fill();
  t->Fill();
  t->Write();
  delete t;
  t = 0;
  f->Close();
  delete f;
  f = 0;
  delete c;
  c = 0;
}

#ifdef TEST
int main()
{
   runwritevararypolyp();
   return 0;
}
#endif

