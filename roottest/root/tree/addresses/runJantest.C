void runJantest() {
  gSystem->Load("libPhysics");
  gROOT->LoadMacro( "JansEvent.C+" );
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("testJan();");
#else
   testJan();
#endif
}
