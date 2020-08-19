{
   gROOT->ProcessLine(".L test_Persistency1.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("test_Persistency1();");
#else
   test_Persistency1();
#endif
}
