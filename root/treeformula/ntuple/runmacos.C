{
   gROOT->ProcessLine(".L macos.C");
   macos("macos");
   new TFile("macos.root");
   Long64_t res = tree->Scan();
   return res==0;
}
