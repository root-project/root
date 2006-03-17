void writefile(int version = 1) {
   gROOT->ProcessLine(Form(".L data%d.C+",version));
   data *a = new data;
   Tdata *b = new Tdata;
   TFile *f = new TFile(Form("data%d.root",version),"RECREATE");
   f->WriteObject(a,"myobj");
   f->WriteObject(b,"myTobj");
   f->Write();
   delete f;
}