// readvararypolyp.C

#include "vararypolyp.C"

#include "TFile.h"
#include "TTree.h"

#include <iostream>

using namespace std;

const char* testfilename = "vararypolyp.root";

void runreadvararypolyp()
{
   C* c = 0;
   TFile* f = new TFile(testfilename);
   f->GetObject("C", c);
   c->print();
   c->clear();
   TTree* t = 0;
   f->GetObject("t", t);
   t->SetBranchAddress("br1.", &c);
   t->GetEntry(1);
   cout << endl;
   c->print();
   delete t;
   t = 0;
   f->Close();
   delete f;
   f = 0;
   delete c;
   c = 0;
}
