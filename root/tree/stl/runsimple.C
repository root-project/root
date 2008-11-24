#include <vector>
#include "TTree.h"

void runsimple() {
   TTree *t = new TTree("T","T");
   std::vector<int> myvec;
   myvec.push_back(3);
   t->Branch("vec",&myvec);
   t->Fill();
   delete t;
}