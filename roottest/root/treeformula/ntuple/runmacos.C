{
   gROOT->ProcessLine(".L macos.C");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("macos(\"macos\");");
#else
   macos("macos");
#endif
   new TFile("macos.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *tree; gFile->GetObject("tree",tree);
#endif
   Long64_t res = tree->Scan();
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(res==0);
#else
   return res==0;
#endif
}
