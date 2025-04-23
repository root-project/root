void Plot(const char*filename) {

   TFile *f1 = new TFile(filename);
   TTree *t1; f1->GetObject("CalTuple",t1);
   if (t1==0) {
      Fatal("runPlot.C : Plot","Missing tree in file %s",filename);
   }
   t1->SetScanField(-1);
   t1->Scan("CalXtalAdcPed[2][0][6][0]:CalXtalAdcPedAllRange[2][0][6][0][1]","CalXtalAdcRng[2][0][6][0]==1");

}

void runPlot(int what=(4|8)) {
   if (what&1) Plot("recon-v1r030603p6_700000811_00000-00984_calTuple.root");
   if (what&2) Plot("recon-v1r030603p6_700000811_02955-03939_calTuple.root");

   if (what&4) {
      TFile *f1 = new TFile("recon-v1r030603p6_700000811_00000-00984_calTuple.root");
      TTree *t1; f1->GetObject("CalTuple",t1);
      TFile *fout = new TFile("cal.root","RECREATE");
      t1->CloneTree(-1,"fast");
      fout->Write();
      fout->Close();
   }

   if (what&8) {
      Plot("cal.root");
   }
}
