void writefile(int version = 1) {
   gROOT->ProcessLine(Form(".L data%d.C+",version));
#if defined(ClingWorkAroundMissingAutoLoading)
   TFile *f;
   f = new TFile(Form("data%d.root",version),"RECREATE");
   gROOT->ProcessLine(TString::Format("TFile *f = (TFile*)%p;\n",f));
   gROOT->ProcessLine("data *a = new data; f->WriteObject(a,\"myobj\");"); 
   gROOT->ProcessLine("Tdata *b = new Tdata;  f->WriteObject(b,\"myTobj\");");
#else
   data *a = new data;
   Tdata *b = new Tdata;
   TFile *f = new TFile(Form("data%d.root",version),"RECREATE");
   f->WriteObject(a,"myobj");
   f->WriteObject(b,"myTobj");
#endif
   f->Write();
   delete f;
}
