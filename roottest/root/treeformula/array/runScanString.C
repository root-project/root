{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L ScanString.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("ScanString();");
#else
   ScanString();
#endif
}
