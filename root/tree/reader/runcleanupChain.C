#include "TFile.h"
#include "TTreeReader.h"
#include "TROOT.h"
#include <vector>
#include <stdio.h>

Bool_t FindObject_loop(TCollection *col, TObject *obj)
{
   TIter iter(col);
   for (TObject *item; (item = iter.Next());) {
      if (item == obj) {
         return kTRUE;
      }
   }
   return kFALSE;
}

TTree *fillTree()
{
   TString name("cleanuptree");
   TTree *tree = new TTree(name, name);
   std::vector<Int_t> myvec;
   for (int i = 0; i < 10; i++)
      myvec.push_back(0x55aa55aa);
   tree->Branch("myvec", &myvec);
   for (int i = 0; i < 10; i++) {
      tree->Fill();
   }
   tree->ResetBranchAddresses();
   return tree;
}

void writefiles()
{
   TFile *output = new TFile("cleanuptree.root", "RECREATE");
   fillTree();
   output->Write();
   delete output;

   TChain myChain("cleanuptree");
   myChain.Add("./cleanuptree.root");
   TFile myFile("cleanupchain.root", "RECREATE");
   myFile.Add(&myChain);
   myFile.Write();
}

int runcleanupChain()
{
   writefiles();

   TChain *chain = 0;
   Bool_t found = kFALSE;

   {
      TFile *myFile = TFile::Open("cleanupchain.root");

      if (!myFile) {
         printf("Could not open test chain root file\n");
         return (1);
      }

      TTreeReader myTreeReader("cleanuptree");
      TTreeReaderValue<std::vector<Int_t>> myvec(myTreeReader, "myvec");

      chain = dynamic_cast<TChain *>(myTreeReader.GetTree());

      if (!chain) {
         printf("Could not get chain from reader\n");
         return (1);
      }

      found = FindObject_loop(gROOT->GetListOfCleanups(), chain);
      printf("(1) Chain %s on list of cleanups\n", found ? "found" : "not found");
      if (!found) {
         cerr << "ERROR!!\n";
         return 1;
      }

      printf("closing root file from which chain was loaded by reader\n");

      delete myFile;

      found = FindObject_loop(gROOT->GetListOfCleanups(), chain);
      printf("(2) Chain %s on list of cleanups\n", found ? "found" : "not found");
      if (!found) {
         cerr << "ERROR!!\n";
         return 1;
      }
   }

   printf("deleted the reader\n");

   found = FindObject_loop(gROOT->GetListOfCleanups(), chain);
   printf("(3) Chain %s on list of cleanups\n", found ? "found" : "not found");
   if (found) {
      cerr << "ERROR!!\n";
      return 1;
   }

   return 0;
}
