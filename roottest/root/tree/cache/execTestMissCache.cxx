#include <TString.h>
#include <TMemFile.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TTreeCache.h>


void execTestMissCache() {

   // create a tree in file

   TFile * in_file = new TMemFile("misscachetester.root", "RECREATE", "", 0);
   in_file->SetCompressionLevel(0);
   in_file->cd();

   TNtuple * in_tree = new TNtuple("in_tree", "blah","a:b:c");
   for (int i = 0; i < 10; ++i) {
      in_tree->Fill(i,2 * i,3 * i);
   }

   in_tree->FlushBaskets();
   for (int i = 0; i < 10; ++i) {
      in_tree->Fill(4 * i, 5 * i, 6 * i);
   }

   in_tree->Write();


   // set cache on input file (mimic TEventIterTree::GetTrees)
   TTreeCache *treeCache = nullptr;
   int cacheSize = 100000;

   TTree *main = static_cast<TTree*>(in_tree);
   TFile *curfile = main->GetCurrentFile();
   main->SetCacheSize(cacheSize);
   treeCache = (TTreeCache *)curfile->GetCacheRead(main);
   if (!treeCache) {
      printf("Error! No tree cache present");
      return;
   }

   // Have no branches in the primary cache at all.
   treeCache->StopLearningPhase();

   float a, b, c;
   main->SetBranchAddress("a", &a);
   main->SetBranchAddress("b", &b);
   main->SetBranchAddress("c", &c);
   Long64_t entries = main->GetEntries();
   for (Long64_t idx=0; idx<entries; idx++) {
      main->GetEntry(idx);
   }

   printf("Disabled miss cache:\n");
   printf("Miss efficiency: %.2f\n", treeCache->GetMissEfficiency());
   printf("Miss relative efficiency: %.2f\n", treeCache->GetMissEfficiencyRel());

   // Re-read cache, enabling the miss cache.
   treeCache->SetOptimizeMisses(true);
   main->SetBranchStatus("c", false);
   for (Long64_t idx=0; idx<entries; idx++) {
      main->GetEntry(idx);
   }

   printf("Enabled miss cache:\n");
   printf("Miss efficiency: %.2f\n", treeCache->GetMissEfficiency());
   printf("Miss relative efficiency: %.2f\n", treeCache->GetMissEfficiencyRel());
}

