{
// Fill out the code of the actual test
   TFile *file = TFile::Open("CaloTowers.root");
   Events->SetScanField(0);
   int n = Events->Scan("CaloTowerCollection.obj.e");
   return !(n==4207);
}
