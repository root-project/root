{
   TFile f("EDM.root");
   Events->SetScanField(0);
   Long64_t res = Events->Scan("CaloDataFrame0.obj.mycell.myBase");
   return res==0;
}