void abcsetup(const char* mode) {
   if (strchr(mode, 'c'))
      gROOT->ProcessLine(".L abc.h+");
   else {
//      gSystem->Exec("genreflex abc.h > /dev/null");
//      gSystem->Exec("g++ -shared -fPIC -I$ROOTSYS/include -L$ROOTSYS/lib -lReflex abc_rflx.cpp -o libabc_rflx.so");
      gSystem->Load("libCintex");
      ROOT::Cintex::Cintex::Enable();
      gSystem->Load("libabc_reflex");
   }
}
