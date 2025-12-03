{
  TFile *_file0 = TFile::Open("AthenaCrossSection.root");
#ifdef ClingWorkAroundMissingDynamicScope
  TTree *CollectionTree; _file0->GetObject("CollectionTree",CollectionTree);
#endif
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate( ! (2 == CollectionTree->Scan("reco_ee_et[][2]:reco_ee_et[0][2]","","",2,0)) );
#else
   return ! (2 == CollectionTree->Scan("reco_ee_et[][2]:reco_ee_et[0][2]","","",2,0));
#endif
}
