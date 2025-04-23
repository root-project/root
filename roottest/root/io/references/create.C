#ifdef ClingWorkAroundUnnamedInclude
#ifndef ClingWorkAroundMissingSmartInclude
#include "A.C+"
#endif
Int_t create() {
#else
{
#endif
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L A.C+");
#else
#ifndef ClingWorkAroundUnnamedInclude
   #include "A.C+"
#endif
#endif
   return !gSystem->CompileMacro("lotsRef.C","kc");
}
