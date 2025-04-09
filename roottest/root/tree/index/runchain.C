#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

bool test(TTree*);

int runchain(){

  ///////////////////////////////////////////////////
  // Make a tree and a file and write them to disk //
  ///////////////////////////////////////////////////

  TFile file("newTestFile.root", "recreate");
  TTree *tree = new TTree("testTree", "my test tree");
  Int_t variable=0;
  tree->Branch("variable", &variable, "variable/I");
  for (int i=0; i<100; i++){
    variable=i;
    tree->Fill();
  }
  tree->Write();
  test(tree);
  file.Close();

  ////////////////////////////////////////////////////
  // Try loading  back in as a TChain               //
  ////////////////////////////////////////////////////
  TChain *chain = new TChain("testTree");
  chain->Add("newTestFile.root");
  bool result = !test(chain);

  delete chain;

  return result;
  
}

bool test(TTree *chain)
{
  cout<<"Entries in chain: "<<chain->GetEntries()<<endl;
  cout<<"BuildIndex returns "<<chain->BuildIndex("variable")<<endl;
  cout<<"Try to get value that is not in the chain, this should return a -1:"<<endl;
  cout<<chain->GetEntryWithIndex(500)<<endl;
  cout<<(int)chain->GetEntryNumberWithIndex(500)<<endl;
  return (chain->GetEntryNumberWithIndex(500)==-1);
}
