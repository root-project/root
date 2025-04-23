////////////////////////////////////////////////////////////////////////
//
//  A simple test program to fill an NtpRecord and write this record
//  to a root tree.
//  The result is a ROOT file "NtpRecord.root" with a tree "Ntp"
//  containing the NtpRecord. The tree has been created with 1 main branch
//  split to level 99.
//
////////////////////////////////////////////////////////////////////////

  #include "TFile.h"
  #include "TTree.h"
  #include "NtpRecord.h"

int FillNtp() {

  TFile* file = new TFile("NtpRecord.root","RECREATE","Ntp Demo file");
  // Create a ROOT Tree and one superbranch
  TTree *tree = new TTree("Ntp","Ntp Demo tree");

  // Tree creation
  NtpRecord* record = new NtpRecord();
  tree->Branch("NtpRecord", "NtpRecord", &record, 16000,99);

  // Generating just one record to fill one entry in tree
  // Fill 3 showers in record
  Int_t nshower = 0;
  TClonesArray& shwarray = *(record -> GetShowers());
  new(shwarray[nshower++])NtpShower(10.);  // argument is shower energy
  new(shwarray[nshower++])NtpShower(20.);
  new(shwarray[nshower++])NtpShower(30.);

  // Fill 4 events in record:
  // event 0 has 2 showers
  // event 1 has 0 showers
  // event 2 has 1 shower
  // event 3 has 0 showers
  Int_t nevent = 0;
  TClonesArray& evtarray = *(record -> GetEvents());
  NtpEvent* event = new(evtarray[nevent++])NtpEvent(0,2); // event,nshower
  event -> GetShowerIndices()[0] = 0;
  event -> GetShowerIndices()[1] = 1;
  new(evtarray[nevent++])NtpEvent(1,0);
  event = new(evtarray[nevent++])NtpEvent(2,1);
  event -> GetShowerIndices()[0] = 2;
  new(evtarray[nevent++])NtpEvent(3,0);

  // Write the lone record to tree  
  tree -> Fill();
  tree -> Write();
  file -> Close();
  delete file; file = 0;

  return 0;
}
