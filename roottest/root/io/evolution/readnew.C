{
gROOT->ProcessLine(".L MyClass.C+");
TFile * file = new TFile("data.root");
MyClass * m = (MyClass*)file->Get("myobj");
m->Dump();
m->check();
TClonesArray * tca = (TClonesArray*)file->Get("collection");
m = (MyClass*)tca->At(0);
m->Dump();
m->check();
m = (MyClass*)tca->At(1);
m->Dump();
m->check();

TClonesArray * tca2 = new TClonesArray("Cont");

TTree * tree = (TTree*)file->Get("T");
tree->SetBranchAddress("conts",&tca2);
tree->SetBranchAddress("objs",&tca);
tree->SetBranchAddress("obj",&m);
tree->GetEntry(0);
m->Dump();
m->check();
m = (MyClass*)tca->At(0);
m->Dump();
m->check();

Cont * c = (Cont*) tca2->At(0);
c->data.Dump();
c = (Cont*) tca2->At(1);
c->data.Dump();

}
