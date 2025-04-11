{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L Enum.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("Enum();");
#else
   Enum();
#endif
}
