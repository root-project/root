{
#ifdef ClingWorkAroundMissingAutoLoading
   gSystem->Load("libTreePlayer");
#endif
   gROOT->ProcessLine(".L all.C+");
   gROOT->ProcessLine(".L write.C");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("write();");
#else
   write();
#endif
}  
