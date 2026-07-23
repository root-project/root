#include "StreamerLoopMemberwise.hxx"

#include "gtest/gtest.h"

#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>

#include <vector>

using namespace ROOTTest::StreamerLoopMemberwise;

// Regression test for the member-wise streaming of a variable-size array
// (`Hit* fHits; //[fN]`, i.e. a TStreamerLoop element) that lives in a base
// class (Frame) at a non-zero offset inside the collection element (Super).
//
// See https://github.com/root-project/root/issues/22895 for more information.
TEST(TStreamerLoopMemberwise, VariableArrayInBaseClass)
{
   const char *fname = "streamerloop_memberwise.root";
   const int kFrames = 4;

   int expectedHits = 0;
   for (int f = 0; f < kFrames; ++f)
      expectedHits += f + 1; // every frame non-empty; n > 0 is what triggered the original bug

   {
      std::vector<Super> slice(kFrames);
      for (int f = 0; f < kFrames; ++f) {
         std::vector<Hit> hits;
         for (int i = 0; i <= f; ++i)
            hits.emplace_back(100 * f + i, 7);
         slice[f].Set(static_cast<int>(hits.size()), hits.data());
      }

      TFile file(fname, "RECREATE");
      TTree tree("T", "T");
      std::vector<Super> *ptr = &slice;
      tree.Branch("slice", &ptr); // default split level -> member-wise collection
      tree.Fill();
      tree.Write();
   }

   int readHits = 0;
   int nullBuffers = 0;
   {
      TFile file(fname);
      auto *tree = file.Get<TTree>("T");
      ASSERT_NE(tree, nullptr);
      std::vector<Super> *slice = nullptr;
      tree->SetBranchAddress("slice", &slice);
      ASSERT_GT(tree->GetEntry(0), 0);
      ASSERT_NE(slice, nullptr);
      ASSERT_EQ(static_cast<int>(slice->size()), kFrames);

      for (int f = 0; f < kFrames; ++f) {
         const Frame &frame = (*slice)[f];
         EXPECT_EQ(frame.fN, f + 1);
         if (frame.fN > 0 && frame.fHits == nullptr)
            ++nullBuffers;
         for (int i = 0; frame.fHits && i < frame.fN; ++i) {
            EXPECT_EQ(frame.fHits[i], Hit(100 * f + i, 7));
            ++readHits;
         }
      }
   }

   EXPECT_EQ(nullBuffers, 0);
   EXPECT_EQ(readHits, expectedHits);

   gSystem->Unlink(fname);
}
