{
   gROOT->ProcessLine(".L ClassConvOld.cxx+");
   TFile *f = new TFile("ClassConv.root","RECREATE");
   TopLevel l(44);
   f->WriteObject(&l,"MyTopLevel");
   f->Write();
   f->Close();
}