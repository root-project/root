{
bool hasEvent = gROOT->GetClass("Event")!=0;
if (hasEvent) {
   TString cmd = gSystem->GetMakeSharedLib();
   cmd.ReplaceAll("$Opt","$Opt -DWITH_EVENT");
   gSystem->SetMakeSharedLib(cmd.Data());
}
gROOT->ProcessLine(".L seold.C+g");
TFile *file = new TFile("Event.new.split9.root");
TTree *tree = (TTree*)file->Get("T");
seold se(tree);
tree->Process(&se);
}
