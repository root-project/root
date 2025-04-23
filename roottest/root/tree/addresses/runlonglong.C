#include "TTree.h"

struct aap {
   ULong64_t A, B, C, D;
};

void runlonglong() {
   TTree* boom = new TTree("b","b");
   aap bokito;
   aap *ptr = &bokito;
   boom->Branch("tak",&bokito,"A/l:B:C:D");
   boom->Branch("takptr.",&ptr);
   bokito.A = 13;
   bokito.B = 14;
   bokito.C = 15;
   bokito.D = 16;
   boom->Fill();
   boom->Show();
}
