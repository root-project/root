{
//new TFile("Event.new.split9.root");
new TFile("Event.new.split2.root");
//new TFile("Event.new.split1.root");
//new TFile("Event.new.split0.root");

tree = (TTree*)gFile->Get("T");
gROOT->ProcessLine(".L GenerateProxy.C+g");
if (gROOT->GetClass("TGenerateProxy")==0) return;
fprintf(stderr,"Will generate gensel.h\n");
TGenerateProxy gp(tree,"script.C","gensel");
//return;
gROOT->ProcessLine(".L gensel.h+");
if (gROOT->GetClass("gensel")==0) return;
gensel s;
tree->Process(&s);
}
