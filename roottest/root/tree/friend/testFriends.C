#include "TChain.h"
#include "TFile.h"

const Int_t noChains = 4;
const char* chainNames[noChains] = {"a", "b", "c", "d"};
const Int_t chainConstatns[noChains] = {1, 2, 3, 4};

const Int_t noTrees = 5;
Int_t size[noTrees] = {100, 150, 200, 120, 30};

void createChain(TString name, Int_t constantValue)
{
   Float_t x, y, z;
   for (Int_t i = 0; i < noTrees; i++) {
      TFile *f = new TFile(Form("%s%d.root",name.Data(),i),"recreate");
      TTree *T = new TTree(name,"test friend trees");
      T->Branch("x", &x, "x/F");
      T->Branch("y", &y, "y/F");
      T->Branch("z", &z, "z/F");
      for (Int_t j = 0; j < ::size[i]; j++) {
         x = j;
         y = ::size[i] - j;
         z = constantValue;
         T->Fill();
      }
      T->Write();
      delete f;
   }
}


void createTrees()
{
   for (Int_t i = 0; i < noChains; i++)
      createChain(chainNames[i], chainConstatns[i]);

}


TChain* chains[noChains];
                                                                                
void testChainFriends(bool recreate = true) {
   if (recreate) createTrees();
   for (Int_t i = 0; i < noChains; i++)
   {
      chains[i] = new TChain(chainNames[i]);
      for (Int_t j = 0; j < noTrees; j++) {
         chains[i]->Add(Form("%s%d.root",chainNames[i],j));
//      chains[i]->Print();
      }
   }
   chains[0]->AddFriend(chains[1],"");
   chains[0]->AddFriend(chains[2],"");
   
   chains[0]->GetEntries();
   //chains[0]->GetTree()->GetListOfFriends()->ls();
   chains[2]->LoadTree(0);
   //chains[0]->GetTree()->GetListOfFriends()->ls();


   chains[0]->Scan("a.z:b.z:c.z","","",3);

   
   chains[2]->AddFriend(chains[3],"");
   chains[2]->Scan("d.z","","",3);

   chains[1]->AddFriend(chains[0],"");

   
   chains[3]->Scan("d.z","","",3);

   chains[2]->Scan("d.z","","",3);
   chains[0]->Scan("a.z:b.z:c.z:d.z","","",3);
   
   chains[2]->Scan("d.z","","",3);
}

//TDSet* dsets[noChains];
//
//void testTDSetFriends() {
//   createTrees();
//   for (Int_t i = 0; i < noChains; i++)
//   {
//      dsets[i] = new TDSet("TTree",chainNames[i]);
//      printf("%s\n", chainNames[i]);
//      for (Int_t j = 0; j < noTrees; j++)
//         dsets[i]->Add(chainNames[i] + j + ".root");
////      chains[i]->Print();
//   }
//   dsets[0]->AddFriend(dsets[1], "");
//   dsets[0]->Draw("b.z:a.x");
//}


void testFriends()
{
   testChainFriends();
//   testTDSetFriends();
}


