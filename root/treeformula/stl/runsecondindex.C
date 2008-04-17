{
  TFile *_file0 = TFile::Open("AthenaCrossSection.root");
  return ! (2 == CollectionTree->Scan("reco_ee_et[][2]:reco_ee_et[0][2]","","",2,0));
}
