#include <TObject.h>
#include <TFile.h>
#include <TTree.h>
#include <TRandom.h>

std::string bname("DST/value"); // broken
//std::string bname("value");   // works


TTree *createTree() {
   TTree *T = new TTree("T", "");

   double val;

   T->Branch(bname.c_str(), &val, "val/D");
   T->Branch("simple",&val);

   for (unsigned i=0; i<5; ++i) {
      val = (i+1) / 13.0;
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
      fprintf(stdout,"%d\t%0.6f\n",i,val);
   }
   
   std::cout << "PRINTING WITH SCAN AGAIN AFTER RENAME\n";
   T->GetBranch(bname.c_str())->SetName("bla");
   T->Scan("val");
   
   delete T;
}

