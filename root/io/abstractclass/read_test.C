{
gROOT->Reset();

gSystem->Load("libData.so");

// create and initialize file
TFile *hfile = TFile::Open("data.root","READ");

DataBlock1 *db1 = (DataBlock1 *) hfile->Get("DataBlock1");
DataBlock2 *db2 = (DataBlock2 *) hfile->Get("DataBlock2");

db1->Print();
db2->Print();

}
