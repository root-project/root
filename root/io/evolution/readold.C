{
gROOT->ProcessLine(".L MyClass.cxx+");
TFile * file = new TFile("data.root");
MyClass * m = (MyClass*)file->Get("myobj");
m->Dump();
m->check();
}
