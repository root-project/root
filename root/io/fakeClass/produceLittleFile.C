{
gROOT->ProcessLine(".L little.C+");
wrapper *e = new wrapper;
TFile *file = new TFile("little.root","RECREATE");
e->Write();
file->Write();
delete file;
}
