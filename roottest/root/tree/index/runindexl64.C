#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

bool test(TTree *);

const char *fname = "indexl64.root";
https : // github.com/root-project/root/pull/19561
        const Long64_t bigval =
           0x0FFFFFFFFFFFFFFF; // here we skip long double, so we can go higher than with runindex64.C

int runindexl64()
{

   ///////////////////////////////////////////////////
   // Make a tree and a file and write them to disk //
   ///////////////////////////////////////////////////

   TFile file(fname, "recreate");
   TTree *tree = new TTree("testTree", "my test tree");
   ULong64_t run, event;
   // ULong64 is "l"
   tree->Branch("run", &run, "run/l");
   tree->Branch("event", &event, "event/l");

   // clang-format off
  ULong64_t   runs[] = { 8,5,5,5,      5, 0,      4, 6, bigval};
  ULong64_t events[] = { 0,1,3,2, bigval, 5, bigval, 3, bigval};
   // clang-format on
   for (size_t i = 0; i < sizeof(events) / sizeof(*events); i++) {
      run = runs[i];
      event = events[i];
      tree->Fill();
   }
   tree->Write();

   bool pass = true;
   cout << "Tree BuildIndex returns " << tree->BuildIndex("run", "event", true, true) << endl;
   for (size_t i = 0; i < sizeof(events) / sizeof(*events); i++) {
      run = runs[i];
      event = events[i];
      pass &= (tree->GetEntryNumberWithIndex(run, event) == i);
   }
   if (!pass) {
      tree->Scan("run:event", "", "colsize=30");
      for (size_t i = 0; i < sizeof(events) / sizeof(*events); i++) {
         run = runs[i];
         event = events[i];
         cout << i << ": Run " << run << ", Event " << event
              << " found at entry number: " << tree->GetEntryNumberWithIndex(run, event) << endl;
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
   cout << "Entries in chain: " << chain->GetEntries() << endl;
   cout << "BuildIndex returns " << chain->BuildIndex("run", "event", true, true) << endl;
   cout << "Try to find the position of run=0, event=500 in the chain, as it does not exist, this should return a -1:"
        << endl;
   cout << chain->GetEntryWithIndex(500) << endl;
   cout << "Try to find the position of run=5, event=bigval in the chain, which was inserted in position 4:" << endl;
   cout << chain->GetEntryNumberWithIndex(5, bigval) << endl;
   return (chain->GetEntryNumberWithIndex(500) == -1) && (chain->GetEntryNumberWithIndex(5, bigval) == 4);
}
