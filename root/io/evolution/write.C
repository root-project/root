{
gROOT->ProcessLine(".L MyClass.cxx+");
TFile * file = new TFile("data.root","RECREATE");
MyClass *m = new MyClass(5);
m->Write("myobj");
file->Write();
file->Close();
}
