#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

bool test(TTree*);

const char* fname = "index64.root";
  // Apple M1 has long double == double; these values exceed its range
  // and cannot be represented as (even temporary) expression results.
  // There would be a warning if you'd try.
  static constexpr bool shortlongdouble = sizeof(longdouble) < 16; // was true for __APPLE__ and __arm64__
  const Long64_t bigval   = shortlongdouble ? 0xFFFFFFFFFFFF : 0xFFFFFFFFFFFFFFF; // still positive number
  const ULong64_t biguval = shortlongdouble ?  0xFFFFFFFFFFFF0 : 0xFFFFFFFFFFFFFFF0; // "negative" number

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

  ULong64_t events[] = { 1,2,3, bigval, biguval, 5 };
  run = 5;
  for(int i=0; i<sizeof(events)/sizeof(*events); i++){
    event = events[i];
    tree->Fill();
  }
  run = 4; event = bigval; tree->Fill();
  run = 6; event = 3; tree->Fill();
  run = biguval; event = bigval; tree->Fill();
  tree->Write();

  cout<<"Tree BuildIndex returns "<<tree->BuildIndex("run", "event")<<endl;
  cout << "Entry should be 3: " << tree->GetEntryNumberWithIndex(5,bigval) << endl;
  cout << "Entry should be 6: " << tree->GetEntryNumberWithIndex(4,bigval) << endl;

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
  cout<<"Try to get value that is not in the chain, this should return a -1:"<<endl;
  cout<<chain->GetEntryWithIndex(500)<<endl;
  cout<<(int)chain->GetEntryNumberWithIndex(5, bigval)<<endl;
  return (chain->GetEntryNumberWithIndex(500)==-1);
}
