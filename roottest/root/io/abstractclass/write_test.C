{
gROOT->Reset();

gSystem->Load("libData.so");

// create and initialize file
TFile *hfile = new TFile("data.root","RECREATE","ROOT file");

DataBlock1 *db1 = new DataBlock1;
DataBlock2 *db2 = new DataBlock2;

db1->Print();
db2->Print();

db1->Write();
db2->Write();

hfile->Write();
hfile->Close();

}
