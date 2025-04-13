void runLHCb(int mode = 1) {
   gSystem->Load("libCintex");
   gROOT->ProcessLine("ROOT::Cintex::Cintex::Enable();");
   if (mode>0) {
      gROOT->ProcessLine(".L classloader.cxx+");  
      pool::RootClassLoader *loader = new pool::RootClassLoader();
   }
   gROOT->ProcessLine(".L libDataModelV1_dictrflx.so");
   f = new TFile("rflx_testv1.root");
   if (!f) {
      cout << "Could not open rflx_testv1.root\n";
      return;
   }
   TTree *tree; f->GetObject("TestTree",tree);
   if (!tree) {
      cout << "Coult not read TestTree from rflx_testv1.root\n";
      return;
   }
   tree->GetEntry(0);
}
