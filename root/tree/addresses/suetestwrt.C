{
// usage: root[0] .x testwrt.C

gSystem -> Load("sueloader_C");
gSystem -> Load("ConfigRecord_cxx");

f = new TFile("configtest.root","RECREATE");
t = new TTree("Config","Config Test");

ConfigRecord* record = 0;
t -> Branch("ConfigRecord","ConfigRecord",&record,32000,99);
// t -> Print();

for ( int i = 0; i < 5; i++ ) {
  Context context(1,2,3);
  RecHeader hdr(context);
  record = new ConfigRecord(hdr);
  t->SetBranchAddress("ConfigRecord",&record); 
  Int_t nbytes = t -> Fill();
  cout << "Filled entry " << i << " with " << nbytes << " bytes." << endl;
  record -> Print();
  delete record; record = 0;
}

t -> Write();
f -> Close();

}


