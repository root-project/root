void macos(char* name)
{
  static Char_t cHead1[9]="123456789"; 
  static UShort_t nCh2=33000;

  //open root file for output
  Char_t rtSuff[6]=".root";
  Char_t rootname[200];
  sprintf(rootname,"%s%s",name,rtSuff);
  cout << "root file: " << rootname << endl;

  TFile *rf=new TFile(rootname,"RECREATE");

  // create a TTree
  TTree *tree=new TTree("tree","test");

  tree->Branch("cHead1",&cHead1,"cHead1[10]/C");
  tree->Branch("nCh2",&nCh2,"nCh2/s");

  tree->Fill();
  rf->Write();
  rf->Close();

}
