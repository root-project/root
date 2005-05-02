{
   TFile *f = TFile::Open("CaloJets.root");
   Events->Scan("CaloJetCollection_CalJt.edm::EDCollection<CaloJet>.obj.data.e","","",1);
   Events->Scan("CaloJetCollection_CalJt.edm::EDCollection<CaloJet>.obj.data.e/2","","",1);
   return 0;
}