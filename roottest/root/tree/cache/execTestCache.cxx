#include "TString.h"
#include <TFile.h>
#include <TTree.h>
#include <TNtuple.h>
#include <TTreeCache.h>


void execTestCache() {
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
   if (fTreeCache) {
      //if (fTreeCache->IsLearning())
      //   Info("GetTrees","the tree cache is in learning phase");
   }



   TFile * out_file = TFile::Open("test_out.root", "RECREATE");
   out_file->SetCompressionLevel(0);
   in_file->cd(); // Probably somewhere inside PROOF code

   TNtuple * out_tree = (TNtuple*) in_tree->CloneTree(0);
   // out_tree->SetName("out_tree");

   //printf("in=%p out=%p\n",in_tree->GetCurrentFile()->GetCacheRead(in_tree),out_tree->GetCurrentFile()->GetCacheRead(out_tree));
   // vvv uncomment the following line to work around the bug
   //~ out_tree->SetDirectory(0);


   out_tree->SetDirectory(out_file);
   if ( in_tree->GetCurrentFile()->GetCacheRead(in_tree) == out_tree->GetCurrentFile()->GetCacheRead(out_tree) ) {
      printf("Error: the file cache are the same!\n");
   }
   out_tree->SetAutoSave();
   out_tree->Fill(-1, -1, -1);


   in_file->Close();

   out_file->cd();
   out_tree->Write();
   out_file->Close();

}


