{
gROOT->Reset();

gSystem->Load("libData");

#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
                   // create and initialize file
"TFile *hfile = TFile::Open(\"data.root\",\"READ\");"
"DataBlock1 *db1 = (DataBlock1 *) hfile->Get(\"DataBlock1\");"
"DataBlock2 *db2 = (DataBlock2 *) hfile->Get(\"DataBlock2\");"
"db1->Print();"
"db2->Print();"
);
#else
   // create and initialize file
   TFile *hfile = TFile::Open("data.root","READ");
   
   DataBlock1 *db1 = (DataBlock1 *) hfile->Get("DataBlock1");
DataBlock2 *db2 = (DataBlock2 *) hfile->Get("DataBlock2");

db1->Print();
db2->Print();
#endif

}
