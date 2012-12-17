{
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   if (1) {
#endif
   gROOT->ProcessLine(".x loadLHCb.C");
   TFile *f = new TFile("lhcb.root");
   gROOT->GetClass("LHCb::ODIN")->GetStreamerInfo()->ls();
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   }
#endif
}
