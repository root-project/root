#include "TTree.h"

struct objs {
   objs() : one(1),two(2) {};
   int one;
   int two;
};

void runaliases(int debug = 0) {
// Fill out the code of the actual test

   TTree *t = new TTree;
   objs o;
   t->Branch("objs",&o,"one/I:two/I");
   t->Fill();
   if (debug) t->Print();
   t->Scan("objs.one");
   t->SetAlias("access","objs");
   t->Scan("access.one");
   t->SetAlias("ones","access.one");
   t->Scan("ones");
   t->SetAlias("add","ones+twos"); 
   t->Scan("add");
   t->SetAlias("twos","objs.two");
   t->Scan("add");
   t->SetAlias("twos","crap");
   t->Scan("add");
   t->SetAlias("crap","add");
   t->Scan("add");
}
