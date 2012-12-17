{
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L A.C+");
#else
   #include "A.C+"
#endif
   return !gSystem->CompileMacro("lotsRef.C","kc");
}
