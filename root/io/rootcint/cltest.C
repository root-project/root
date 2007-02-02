#include "cltestClass.cxx"
#include "cltestLinkdef.h"
#include "TFile.h"

void cltest() 
{
   TFile *f = TFile::Open("cltest.root","RECREATE");
   TestClass *c = new TestClass();
   c->Write();
   delete f;
}
