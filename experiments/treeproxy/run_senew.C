{
gROOT->ProcessLine(".L TProxy.h+g");
if (gROOT->GetClass("TProxy")==0) return;


bool hasEvent = gROOT->GetClass("Event")!=0;
if (hasEvent) {
   TString cmd = gSystem->GetMakeSharedLib();
   cmd.ReplaceAll("$Opt","$Opt -DWITH_EVENT");
   gSystem->SetMakeSharedLib(cmd.Data());
}

gROOT->ProcessLine(".L senew.C+g");
if (gROOT->GetClass("senew")==0) return;
TFile *file = new TFile("Event.new.split9.root");
TTree *tree = (TTree*)file->Get("T");
senew se(tree);
tree->Process(&se);
}
