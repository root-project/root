{
#ifdef ClingWorkAroundMissingSmartInclude
   gROOT->ProcessLine(".L missingsplit_read.C+");
#else
   #include "missingsplit_read.C+"
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("missingsplit_read();");
#else
   missingsplit_read();
#endif
}
