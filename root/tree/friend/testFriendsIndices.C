// If everthing is ok the result should be a plot with a single straight line y=x.


#include <algorithm>
#include <random>
#include <vector>
#include "TFile.h"
#include "TChain.h"
#include "TError.h"

const Int_t noChains = 2;
const char* chainNames[noChains] = {"e", "f"};
const Float_t chainConstants[noChains] = {1, 2};

const Int_t noTrees = 5;
Int_t size[noTrees] = {100, 150, 200, 120, 30};  // real sizes of the trees is 2x these values
using namespace std;

 


void createFiles(TString name, Float_t valConst)
{
   // creates a chains having noTrees trees with the 2*sizes from size array
   // each has branches: val, val2 ( = val*val and val*val+1),
   // val3 ( = val + val2 + valCost) and valConst ( = valConst)

   Int_t nextToFill = 1;
   Int_t val;
   Int_t val2;
   Int_t val3;
   std::mt19937 urng;
   for (int treeNo = 0; treeNo < noTrees; treeNo++) {
      vector<Int_t> data;
      for (int j = 0; j < ::size[treeNo]; j++) {
         data.push_back(nextToFill++);
      }
      shuffle(data.begin(), data.end(), urng);

      TString filename;
      filename += name;
      filename += treeNo;
      filename += ".root";
      TFile *f = new TFile(filename,"recreate");
      TTree *T = new TTree(name,"test friend trees and indices");
      printf("Creating %s\n", filename.Data());
      T->Branch("val", &val, "val/I");
      T->Branch("val2", &val2, "val2/I");
      T->Branch("val3", &val3, "val3/I");
      T->Branch("valConst", &valConst, "valConst/F");

      for (Int_t j = 0; j < ::size[treeNo]; j++) {
         val = data[j];
         val2 = val*val;
         val3 = val + val2 + (Int_t)valConst;
         T->Fill();

         val = data[j];
         val2 = val*val + 1;
         val3 = val + val2 + (Int_t)valConst;
         T->Fill();
      }
      if (treeNo % 2 == 1) T->BuildIndex("val", "val2");
      T->Write();
      delete f;
   }
}

void create()
{
   for (Int_t i = 0; i < noChains; i++)
      createFiles(chainNames[i], chainConstants[i]);
}


TChain* chains[noChains];

Bool_t gVerbose = kTRUE;

void testChainFriendsWithIndex(int what = 3) {
   if (what & 4) gVerbose = kFALSE;
   if (what & 1) create();
   if (what & 2) {
      for (Int_t i = 0; i < noChains; i++)
         {
            chains[i] = new TChain(chainNames[i]);
            for (Int_t j = 0; j < noTrees; j++) {
               TString filename;
               filename += chainNames[i];
               filename += j;
               filename += ".root";
               chains[i]->Add(filename);
            }
            chains[i]->BuildIndex("val", "val2");
            //      chains[i]->Print();
         }
      chains[1]->BuildIndex("val", "val2");
      chains[0]->AddFriend(chains[1],"");
      if (gVerbose) chains[0]->Scan("f.val:e.val:f.val-e.val","","colsize=13",20);
      else chains[0]->Scan("f.val-e.val","","colsize=13",20);
   }
}

/*
TDSet* dsets[noChains];

void testTDSetFriends() {
   createTrees();
   for (Int_t i = 0; i < noChains; i++)
   {
      dsets[i] = new TDSet("TTree",chainNames[i]);
      printf("%s\n", chainNames[i]);
      for (Int_t j = 0; j < noTrees; j++) {
         TString filename;
         filename += chainNames[i];
         filename += j;
         filename += ".root";
         dsets[i]->Add(filename);
      }
//      chains[i]->Print();
   }
   dsets[0]->AddFriend(dsets[1], "");
   dsets[0]->AddFriend(dsets[2], "");
   dsets[0]->AddFriend(dsets[3], "");
   dsets[0]->Draw("d.z:a.x");
}
*/
void testFriendsIndices(int what = 3)
{
   testChainFriendsWithIndex(what);
//   testTDSetFriends();
}


