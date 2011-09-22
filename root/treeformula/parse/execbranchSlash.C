#include <TObject.h>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>

std::string bname("DST/value"); // broken
//std::string bname("value");   // works


TTree *createTree() {
   TTree *T = new TTree("T", "");

   double val;
   gRandom->SetSeed(0);

   T->Branch(bname.c_str(), &val, "val/D");
   T->Branch("simple",&val);

   for (unsigned i=0; i<5; ++i) {
      val = gRandom->Rndm();
      T->Fill();
   }
   T->ResetBranchAddresses();
   return T;
}

void execbranchSlash() 
{
   TTree *T = createTree();
   std::cout << "PRINTING WITH SCAN\n";
   T->Scan("val");
   
   std::cout << "PRINTING WITH TTree::GetEntry\n";
   double val;
   T->SetBranchAddress(bname.c_str(), &val);
   for (int i = 0; i<T->GetEntries(); i++) {
      T->GetEntry(i);
      std::cout << i <<'\t'<< val << std::endl;
   }
   
   std::cout << "PRINTING WITH SCAN AGAIN AFTER RENAME\n";
   T->GetBranch(bname.c_str())->SetName("bla");
   T->Scan("val");
   
   delete T;
}

