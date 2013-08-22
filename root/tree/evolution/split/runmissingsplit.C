#ifdef ClingWorkAroundUnnamedInclude
#ifndef ClingWorkAroundMissingSmartInclude
#include "missingsplit_read.C+"
#endif
void runmissingsplit() {
#else
{
#endif
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L missingsplit_read.C+");
#else
#ifndef ClingWorkAroundUnnamedInclude
   #include "missingsplit_read.C+"
#endif
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("missingsplit_read();");
#else
   missingsplit_read();
#endif
}
