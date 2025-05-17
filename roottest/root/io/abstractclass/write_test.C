#include "DataBlock1.h"
#include "DataBlock2.h"

void write_test()
{
   gSystem->Load("libAbstractClasses");

   // create and initialize file
   auto hfile = TFile::Open("data.root", "RECREATE");

   DataBlock1 *db1 = new DataBlock1;
   DataBlock2 *db2 = new DataBlock2;

   db1->Print();
   db2->Print();

   db1->Write();
   db2->Write();

   hfile->Write();
   hfile->Close();
}
