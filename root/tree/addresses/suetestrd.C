{
// usage: root[0] .x testrd.C

gSystem -> Load("sueloader_C");
gSystem -> Load("ConfigRecord_cxx");

f = new TFile("configtest.root","READ");
t = (TTree*)(f->Get("Config"));

ConfigRecord* record = 0;

for ( int i = 0; i < 5; i++ ) {
  t -> SetBranchAddress("ConfigRecord",&record);
  Int_t nbytes = t -> GetEntry(i);
  cout << "Retrieved entry " << i << " with " << nbytes << " bytes." << endl;
  record -> Print();
  delete record; record = 0;
}

f -> Close();

}


