#ifdef ClingWorkAroundMissingAutoLoading
class TBranch;
class TBranchRef;
#endif
#include "TFile.h"
#include "TTree.h"

int runvectorInVector() {
// Fill out the code of the actual test
   TFile *file = TFile::Open("CaloTowers.root");
   gSystem->Load("libTreePlayer");
   TTree *Events; file->GetObject("Events",Events);
   Events->SetScanField(0);

   //TBranch *b = Events->GetBranch("obj.layers");
   //b->GetEntry(0);
   //gDebug = 8;
   //b->GetEntry(1);

   Long64_t n;
   n = Events->Scan("CaloTowerCollection.obj.e");
   if (n!=4207) { return 1; }
   n = Events->Scan("CaloTowerCollection.obj.layers.e");
#ifdef ClingWorkAroundErracticValuePrinter
   printf("(int)0\n");
#endif
   return (n!=3128);
}
