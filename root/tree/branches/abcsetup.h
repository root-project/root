void abcsetup(const char* mode) {
   if (strchr(mode, 'c'))
      gROOT->ProcessLine(".L abc.h+");
   else {
      gSystem->Load("libCintex");
      ROOT::Cintex::Cintex::Enable();
      gROOT->ProcessLine(".L libabc_rflx.C+");
   }
}
