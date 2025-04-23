{
   gROOT->ProcessLine(".L cms.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("CMSTestRead();");
#else
   CMSTestRead();
#endif
}
