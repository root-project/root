void abcsetup(const char* mode) {
   if (strchr(mode, 'c'))
#ifndef ClingWorkAroundMissingDynamicScope
      gROOT->ProcessLine(".L abc.h+");
#else 
      ;
#endif
   else {
      gSystem->Load("libCintex");
#ifdef ClingWorkAroundMissingDynamicScope
      gROOT->ProcessLine("ROOT::Cintex::Cintex::Enable();");
#else
      ROOT::Cintex::Cintex::Enable();
#endif
      gROOT->ProcessLine(".L libabc_rflx.C+");
   }
}
