{
   TFile *f = TFile::Open("CaloJets.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *Events; f->GetObject("Events",Events);
#endif
   Events->Scan("CaloJetCollection_CalJt.edm::EDCollection<CaloJet>.obj.data.e","","",1);
   Events->Scan("CaloJetCollection_CalJt.edm::EDCollection<CaloJet>.obj.data.e/2","","",1);
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}