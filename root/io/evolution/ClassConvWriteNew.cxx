{
   gROOT->ProcessLine(".L ClassConvNew.cxx+");
   TFile *f = new TFile("ClassConvNew.root","RECREATE");
   TopLevel l(44);
   f->WriteObject(&l,"MyTopLevel");
   f->Write();
   f->Close();
}
