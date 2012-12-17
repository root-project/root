{
   gROOT->ProcessLine(".L ClassConvNew.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
      "TFile *f = new TFile(\"ClassConvNew.root\",\"RECREATE\");"
      "TopLevel l(44);"
      "f->WriteObject(&l,\"MyTopLevel\");"
      "f->Write();"
      "f->Close();");
#else
   TFile *f = new TFile("ClassConvNew.root","RECREATE");
   TopLevel l(44);
   f->WriteObject(&l,"MyTopLevel");
   f->Write();
   f->Close();
#endif
}
