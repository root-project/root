void reflexrun() {
   gSystem->Load("libCintex");
   Cintex::Enable();
   gSystem->Load("libt_rflx");
   gROOT->ProcessLine(".x cintrun.C");
}
