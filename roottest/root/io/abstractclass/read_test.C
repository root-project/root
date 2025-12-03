#include "DataBlock1.h"
#include "DataBlock2.h"

void read_test(const char *fname = "data.root")
{
   gSystem->Load("libAbstractClasses");

   // create and initialize file
   auto hfile = TFile::Open(fname,"READ");

   DataBlock1 *db1 = (DataBlock1 *) hfile->Get("DataBlock1");
   DataBlock2 *db2 = (DataBlock2 *) hfile->Get("DataBlock2");

   db1->Print();
   db2->Print();

}
