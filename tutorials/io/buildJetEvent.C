void buildJetEvent()
{
   TString tutdir = gROOT->GetTutorialDir();
   gROOT->ProcessLine(".include " + tutdir + "/tree");
   gROOT->ProcessLine(".L " + tutdir + "/tree/JetEvent.cxx+");
}
