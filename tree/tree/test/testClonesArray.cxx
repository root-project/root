#include "TObject.h"
#include "TestRefObj.h"
#include "TFile.h"
#include "TTree.h"
#include "TRandom.h"
#include "TClonesArray.h"
#include "TProcessID.h"
#include "TSystem.h"

#include "gtest/gtest.h"

// https://its.cern.ch/jira/browse/ROOT-7249
TEST(TClonesArray, RefArrayClearChildren)
{
   Bool_t resetObjectCount = kTRUE;
   Bool_t activateBranchRef = kFALSE;
   Int_t splitLevel = 1;
   Bool_t pruneSecondChildren = kTRUE;
   auto filename = "test7249.root";
   auto treename = "tree";
   // streamwithrefs
   {
      TFile testFile(filename, "RECREATE");
      TTree dataTree(treename, treename);

      TClonesArray particles(TestRefObj::Class(), 100);
      TClonesArray children(TestRefObj::Class(), 100);

      dataTree.Branch("particles", &particles, 32768, splitLevel);
      dataTree.Branch("children", &children, 32768, splitLevel);
      if (activateBranchRef) {
         dataTree.BranchRef();
      }

      for (Int_t e = 0; e < 1000; e++) {
         // For each "event".
         UInt_t objCount = TProcessID::GetObjectCount();

         TestRefObj *motherPart = static_cast<TestRefObj *>(particles.ConstructedAt(0));
         TestRefObj *childPart = static_cast<TestRefObj *>(children.ConstructedAt(0));
         motherPart->SetChild(childPart);

         // Prune all children if requested and event odd.
         if (pruneSecondChildren && (e % 2 != 0)) {
            children.Clear("C");
         }

         dataTree.Fill();

         children.Clear("C");
         particles.Clear("C");

         if (resetObjectCount) {
            TProcessID::SetObjectCount(objCount);
         }
      }
      dataTree.Write();
      testFile.Close();
   }
   // readwithrefs
   {
      TFile testFile(filename, "READ");
      TTree *dataTree = testFile.Get<TTree>(treename);

      TClonesArray *particles = nullptr;
      TClonesArray *children = nullptr;

      dataTree->SetBranchAddress("particles", &particles);
      dataTree->SetBranchAddress("children", &children);

      if (activateBranchRef) {
         dataTree->BranchRef();
      }
      const Long64_t entries = dataTree->GetEntriesFast();
      for (Long64_t e = 0; e < entries; e++) {
         dataTree->GetEntry(e);

         // For each "event".
         UInt_t objCount = TProcessID::GetObjectCount();

         // UInt_t parts = particles->GetEntries();
         UInt_t childs = children->GetEntries();

         auto parti = static_cast<TestRefObj *>(particles->UncheckedAt(0));
         if (pruneSecondChildren && (e % 2 != 0)) {
            ASSERT_EQ(childs, 0);
            ASSERT_FALSE(parti->HasChild());
         } else {
            ASSERT_NE(childs, 0);
            ASSERT_TRUE(parti->HasChild());
         }

         children->Clear("C");
         particles->Clear("C");

         if (resetObjectCount) {
            TProcessID::SetObjectCount(objCount);
         }
      }

      delete dataTree;
      testFile.Close();
   }
   gSystem->Unlink(filename);
}

// https://its.cern.ch/jira/browse/ROOT-7473
TEST(TClonesArray, ClearSlot)
{
   TClonesArray particles(TestRefObj::Class(), 100);
   TClonesArray children(TestRefObj::Class(), 100);
   TestRefObj *motherPart = static_cast<TestRefObj *>(particles.ConstructedAt(0));
   TestRefObj *childPart = static_cast<TestRefObj *>(children.ConstructedAt(0));
   motherPart->SetChild(childPart);
   particles.ClearSlot(0, "C");
   auto parti = static_cast<TestRefObj *>(particles.UncheckedAt(0));
   ASSERT_EQ(parti, nullptr);
   auto child = static_cast<TestRefObj *>(children.UncheckedAt(0));
   ASSERT_NE(child, nullptr);
}
