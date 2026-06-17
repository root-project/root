
#include "ntuple_test.hxx"

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>

#include <vector>
#include <string>
#include <numeric>

#include <utility>
#include <cstdio>

using ROOT::Experimental::RNTupleOpenSpec;
using ROOT::Experimental::RNTupleProcessor;

class GH16805ProcessorTest : public testing::Test {
protected:
   const std::vector<std::string> fStepZeroFiles{
      "gh16805_rntuple_stepzero_0.root",
      "gh16805_rntuple_stepzero_1.root"
   };

   const std::vector<std::string> fFriendFiles{
      "gh16805_rntuple_friend_0.root",
      "gh16805_rntuple_friend_1.root",
      "gh16805_rntuple_friend_2.root"
   };

   const std::string fStepOneFile = "gh16805_rntuple_stepone.root";

   void WriteStepZero(const std::string &fileName, int begin, int end)
   {
      auto model = ROOT::RNTupleModel::Create();

      auto br1 = model->MakeField<int>("stepZeroBr1");
      auto br2 = model->MakeField<int>("stepZeroBr2");

      auto writer =
         ROOT::RNTupleWriter::Recreate(std::move(model), "stepzero", fileName);

      for (int i = begin; i < end; ++i) {
         *br1 = i;
         *br2 = 2 * i;
         writer->Fill();
      }
   }

   void WriteStepOne(const std::string &fileName, int begin, int end)
   {
      auto model = ROOT::RNTupleModel::Create();

      auto br1 = model->MakeField<int>("stepOneBr1");

      auto writer =
         ROOT::RNTupleWriter::Recreate(std::move(model), "stepone", fileName);

      for (int i = begin; i < end; ++i) {
         *br1 = i;
         writer->Fill();
      }
   }

   void WriteFriend(const std::string &fileName, int begin, int end)
   {
      auto model = ROOT::RNTupleModel::Create();

      auto br1 = model->MakeField<int>("friendBr1");
      auto br2 = model->MakeField<int>("friendBr2");

      auto writer =
         ROOT::RNTupleWriter::Recreate(std::move(model),
                                       "topLevelFriend",
                                       fileName);

      for (int i = begin; i < end; ++i) {
         *br1 = i;
         *br2 = 2 * i;
         writer->Fill();
      }
   }

   void SetUp() override
   {
      WriteStepZero(fStepZeroFiles[0], 0, 10);
      WriteStepZero(fStepZeroFiles[1], 10, 20);

      WriteFriend(fFriendFiles[0], 200, 207);
      WriteFriend(fFriendFiles[1], 207, 214);
      WriteFriend(fFriendFiles[2], 214, 220);

      WriteStepOne(fStepOneFile, 100, 120);
   }

   void TearDown() override
   {
      for (const auto &f : fStepZeroFiles)
         std::remove(f.c_str());

      for (const auto &f : fFriendFiles)
         std::remove(f.c_str());

      std::remove(fStepOneFile.c_str());
   }

};

TEST_F(GH16805ProcessorTest, JoinReading)
{
   std::vector<RNTupleOpenSpec> stepOneSpecs{
      {"stepone", fStepOneFile}
   };

   std::vector<RNTupleOpenSpec> stepZeroSpecs{
      {"stepzero", fStepZeroFiles[0]},
      {"stepzero", fStepZeroFiles[1]}
   };

   std::vector<RNTupleOpenSpec> friendSpecs{
      {"topLevelFriend", fFriendFiles[0]},
      {"topLevelFriend", fFriendFiles[1]},
      {"topLevelFriend", fFriendFiles[2]}
   };

   auto stepOneProc =
      RNTupleProcessor::CreateChain(stepOneSpecs, "stepone");

   auto stepZeroProc =
      RNTupleProcessor::CreateChain(stepZeroSpecs, "stepzero");

   auto friendProc =
      RNTupleProcessor::CreateChain(friendSpecs, "topLevelFriend");

   auto joinedWithFriend =
      RNTupleProcessor::CreateJoin(
         std::move(stepOneProc),
         std::move(friendProc),
         {}
      );

   auto joinedAll =
      RNTupleProcessor::CreateJoin(
         std::move(joinedWithFriend),
         std::move(stepZeroProc),
         {}
      );

   auto stepOneBr1 = joinedAll->RequestField<int>("stepOneBr1");
   auto friendBr1 = joinedAll->RequestField<int>("topLevelFriend.friendBr1");
   auto friendBr2 = joinedAll->RequestField<int>("topLevelFriend.friendBr2");
   auto stepZeroBr1 = joinedAll->RequestField<int>("stepzero.stepZeroBr1");
   auto stepZeroBr2 = joinedAll->RequestField<int>("stepzero.stepZeroBr2");

   std::size_t i = 0;

   for (auto idx : *joinedAll) {
      EXPECT_EQ(i, idx);

      EXPECT_EQ(static_cast<int>(i), *stepZeroBr1);
      EXPECT_EQ(static_cast<int>(2 * i), *stepZeroBr2);
      EXPECT_EQ(static_cast<int>(100 + i), *stepOneBr1);
      EXPECT_EQ(static_cast<int>(200 + i), *friendBr1);
      EXPECT_EQ(static_cast<int>(2 * (200 + i)), *friendBr2);

      ++i;
   }

   EXPECT_EQ(20u, i);
   EXPECT_EQ(20u, joinedAll->GetNEntriesProcessed());
}
