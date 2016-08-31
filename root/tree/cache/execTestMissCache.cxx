#include "TString.h"
#include <TFile.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TTreeCache.h>


void execTestMissCache() {
   TString in_fname("test_in.root");

   // create a tree in file
   {
      TFile * in_file = TFile::Open(in_fname, "RECREATE");
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
      in_file->Close();
   }

   TFile * in_file = TFile::Open(in_fname);

   TNtuple * in_tree = (TNtuple*) in_file->Get("in_tree");

   // set cache on input file (mimic TEventIterTree::GetTrees)
   TTreeCache *fTreeCache = 0;
   int fCacheSize = 100000;

   TTree * main = (TTree*) in_tree;
   TFile *curfile = main->GetCurrentFile();
   if (!fTreeCache) {
      main->SetCacheSize(fCacheSize);
      fTreeCache = (TTreeCache *)curfile->GetCacheRead(main);
      if (fCacheSize < 0) fCacheSize = main->GetCacheSize();
   } else {
      curfile->SetCacheRead(fTreeCache, main);
      fTreeCache->UpdateBranches(main);
   }
   if (!fTreeCache) {
      printf("Error! No tree cache present");
      return;
   }

   // Have no branches in the primary cache at all.
   fTreeCache->StopLearningPhase();

   float a, b, c;
   main->SetBranchAddress("a", &a);
   main->SetBranchAddress("b", &b);
   main->SetBranchAddress("c", &c);
   Long64_t entries = main->GetEntries();
   for (Long64_t idx=0; idx<entries; idx++) {
      main->GetEntry(idx);
   }

   printf("Disabled miss cache:\n");
   printf("Miss efficiency: %.2f\n", fTreeCache->GetMissEfficiency());
   printf("Miss relative efficiency: %.2f\n", fTreeCache->GetMissEfficiencyRel());

   // Re-read cache, enabling the miss cache.
   fTreeCache->SetOptimizeMisses(true);
   main->SetBranchStatus("c", false);
   for (Long64_t idx=0; idx<entries; idx++) {
      main->GetEntry(idx);
   }

   printf("Enabled miss cache:\n");
   printf("Miss efficiency: %.2f\n", fTreeCache->GetMissEfficiency());
   printf("Miss relative efficiency: %.2f\n", fTreeCache->GetMissEfficiencyRel());
}

