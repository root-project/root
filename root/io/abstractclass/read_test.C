{
gROOT->Reset();

#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
if (1) {
#endif
gSystem->Load("libData");

// create and initialize file
TFile *hfile = TFile::Open("data.root","READ");

#ifdef ClingWorkAroundMissingDynamicScope
gROOT->ProcessLine(
"DataBlock1 *db1 = (DataBlock1 *) hfile->Get(\"DataBlock1\");"
"DataBlock2 *db2 = (DataBlock2 *) hfile->Get(\"DataBlock2\");"
"db1->Print();"
"db2->Print();"
);
#else
DataBlock1 *db1 = (DataBlock1 *) hfile->Get("DataBlock1");
DataBlock2 *db2 = (DataBlock2 *) hfile->Get("DataBlock2");

db1->Print();
db2->Print();
#endif

#ifdef ClingWorkAroundUnnamedIncorrectInitOrder

#endif
}
