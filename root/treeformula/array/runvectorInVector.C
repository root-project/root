{
// Fill out the code of the actual test
   TFile *file = TFile::Open("CaloTowers.root");
   Events->SetScanField(0);
   Long64_t n = Events->Scan("CaloTowerCollection.obj.e");
   if (n!=4207) { return 1; }
   n = Events->Scan("CaloTowerCollection.obj.layers.e");
   return (n!=3128);
}
