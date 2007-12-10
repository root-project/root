{
   gROOT->ProcessLine(".x loadLHCb.C");
   TFile *f = new TFile("lhcb.root");
   gROOT->GetClass("LHCb::ODIN")->GetStreamerInfo()->ls();
}
