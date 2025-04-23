{
#ifdef ClingWorkAroundMissingAutoLoading
   gSystem->Load("libTreePlayer");
#endif
   gROOT->ProcessLine(".L all.C+");
#ifdef ClingWorkAroundCallfuncAndInline
   gROOT->ProcessLine(".L write.C+");
#else
   gROOT->ProcessLine(".L write.C");
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("write();");
#else
   write();
#endif
}  
