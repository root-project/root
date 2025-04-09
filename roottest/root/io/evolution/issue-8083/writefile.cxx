R__LOAD_LIBRARY(stringarray_h)
#include "stringarray.h"
#include "TFile.h"
#include "TTree.h"

void writefile () 
{ 
auto f = new TFile("stringarray.root", "RECREATE");
auto t = new TTree("t", "t");
mystrarray arr;
std::vector<mystrarray> vec;
container c;
t->Branch("obj.", &arr);
t->Branch("vec.", &vec);
t->Branch("cont.", &c);
t->Print();
t->Fill();
f->Write();
}
