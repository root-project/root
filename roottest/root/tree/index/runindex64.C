#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

bool test(TTree*);

const char* fname = "index64.root";
// Apple M1 has long double == double; these values exceed its range
// and cannot be represented as (even temporary) expression results.
// There would be a warning if you'd try.
// More info: https://github.com/root-project/roottest/commit/f3c97809c9064feccaed3844007de9e7c6a5980d and https://github.com/root-project/roottest/commit/9e3843d4bf50bc34e6e15dfe7c027f029417d6c0
// static constexpr bool shortlongdouble = sizeof(long double) < 16; // was true for __APPLE__ and __arm64__
// const Long64_t bigval   = shortlongdouble ?  0x0FFFFFFFFFFFF : 0x0FFFFFFFFFFFFFFF; // still positive number
// const ULong64_t biguval = shortlongdouble ?  0xFFFFFFFFFFFF0 : 0xFFFFFFFFFFFFFFF0; // "negative" number
const Long64_t bigval = 0xFFFFFFFFFFFF0; // larger values fail on __APPLE__ / __arm64__ because the long double is less than 16 bytes.
// const ULong64_t biguval = bigval;

int runindex64(){

  ///////////////////////////////////////////////////
  // Make a tree and a file and write them to disk //
  ///////////////////////////////////////////////////

  TFile file(fname, "recreate");
  TTree *tree = new TTree("testTree", "my test tree");
  ULong64_t     run, event;
  // ULong64 is "l"
  tree->Branch("run", &run, "run/l");
  tree->Branch("event", &event, "event/l");

  ULong64_t   runs[] = { 8,5,5,5,      5, 0,      4, 6, bigval};
  ULong64_t events[] = { 0,1,3,2, bigval, 5, bigval, 3, bigval};
  for(size_t i=0; i<sizeof(events)/sizeof(*events); i++){
    run = runs[i];
    event = events[i];
    tree->Fill();
  }
  tree->Write();
  
  bool pass = true;
  cout<<"Tree BuildIndex returns "<<tree->BuildIndex("run", "event")<<endl;
  for (size_t i=0; i<sizeof(events)/sizeof(*events); i++) {
    run = runs[i];
    event = events[i];
    pass &= (tree->GetEntryNumberWithIndex(run, event) == i);
  }
  if (!pass) {
    tree->Scan("run:event","","colsize=30");
    for (size_t i=0; i<sizeof(events)/sizeof(*events); i++) {
      run = runs[i];
      event = events[i];
      cout << i << ": Run " << run << ", Event " << event << " found at entry number: " << tree->GetEntryNumberWithIndex(run, event) << endl;
    }
  }

  test(tree);
  file.Close();

  ////////////////////////////////////////////////////
  // Try loading  back in as a TChain               //
  ////////////////////////////////////////////////////
  TChain *chain = new TChain("testTree");
  chain->Add(fname);
  bool result = !test(chain);

  delete chain;

  return result;
  
}

bool test(TTree *chain)
{
  cout<<"Entries in chain: "<<chain->GetEntries()<<endl;
  cout<<"BuildIndex returns "<<chain->BuildIndex("run", "event")<<endl;
  cout<<"Try to find the position of run=0, event=500 in the chain, as it does not exist, this should return a -1:"<<endl;
  cout<<chain->GetEntryWithIndex(500)<<endl;
  cout<<"Try to find the position of run=5, event=bigval in the chain, which was inserted in position 4:"<<endl;
  cout<<chain->GetEntryNumberWithIndex(5, bigval)<<endl;
  return (chain->GetEntryNumberWithIndex(500)==-1) && (chain->GetEntryNumberWithIndex(5, bigval) == 4);
}
