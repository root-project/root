{
new TFile("Event.new.split9.root");
//new TFile("Event.new.split1.root");

tree = (TTree*)gFile->Get("T");
gROOT->ProcessLine(".L GenerateProxy.C+g");
if (gROOT->GetClass("TGenerateProxy")==0) return;
TGenerateProxy gp(tree,"script.C","gensel");
gROOT->ProcessLine(".L gensel.h+");
if (gROOT->GetClass("gensel")==0) return;
gensel s;
tree->Process(&s);
}
