{
// Fill out the code of the actual test
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L A.C+");
#endif
   gROOT->ProcessLine(".x lotsRef.C(0)");
}
