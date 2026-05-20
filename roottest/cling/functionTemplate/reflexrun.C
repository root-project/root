void reflexrun() {
   gSystem->Load("libCintex");
   Cintex::Enable();
   gSystem->Load("t_rflx_wrap_cxx");
   gROOT->ProcessLine(".x cintrun.C");
}
