{
   gROOT->ProcessLine(".L ClassConvOld.cxx+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(
      "TFile *f = new TFile(\"ClassConv.root\",\"RECREATE\");"
      "TopLevel l(44);"
      "f->WriteObject(&l,\"MyTopLevel\");"
      "f->Write();"
      "f->Close();")
#else
   TFile *f = new TFile("ClassConv.root","RECREATE");
   TopLevel l(44);
   f->WriteObject(&l,"MyTopLevel");
   f->Write();
   f->Close();
#endif
}