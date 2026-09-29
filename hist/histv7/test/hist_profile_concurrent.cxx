#include "hist_test.hxx"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

TEST(RProfileConcurrentFiller, Constructor)
{
   static constexpr std::size_t Bins = 20;
   auto profile = std::make_shared<RProfile>(Bins, std::make_pair(0, Bins));
   RProfileConcurrentFiller filler(profile);

   std::shared_ptr<RProfile> profilePtr = filler.GetProfile();
   EXPECT_EQ(profile, profilePtr);

   auto context = filler.CreateFillContext();
   context->Flush();

   EXPECT_THROW(RProfileConcurrentFiller(nullptr), std::invalid_argument);
}

TEST(RProfileConcurrentFiller, OldEntries)
{
   static constexpr std::size_t Bins = 20;
   auto profile = std::make_shared<RProfile>(Bins, std::make_pair(0, Bins));
   profile->Fill(8.5, 23.0);
   ASSERT_EQ(profile->GetNEntries(), 1);
   ASSERT_EQ(profile->GetBinContent(8).fSum, 1.0);

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      context->Flush();
   }

   EXPECT_EQ(profile->GetNEntries(), 1);
   EXPECT_EQ(profile->GetBinContent(8).fSum, 1.0);
}

TEST(RProfileFillContext, Fill)
{
   static constexpr std::size_t Bins = 20;
   auto profile = std::make_shared<RProfile>(Bins, std::make_pair(0, Bins));

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      context->Fill(8.5, 23.0);
      context->Fill(std::make_tuple(9.5), 25.0);
   }

   auto &bin8 = profile->GetBinContent(RBinIndex(8));
   EXPECT_EQ(bin8.fSumValues, 23.0);
   EXPECT_EQ(bin8.fSumValues2, 529.0);
   EXPECT_EQ(bin8.fSum, 1.0);
   EXPECT_EQ(bin8.fSum2, 1.0);
   std::array<RBinIndex, 1> indices = {9};
   auto &bin9 = profile->GetBinContent(indices);
   EXPECT_EQ(bin9.fSumValues, 25.0);
   EXPECT_EQ(bin9.fSumValues2, 625.0);
   EXPECT_EQ(bin9.fSum, 1.0);
   EXPECT_EQ(bin9.fSum2, 1.0);

   EXPECT_EQ(profile->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile->ComputeNEffectiveEntries(), 2);
   EXPECT_FLOAT_EQ(profile->ComputeMean(0), 9);
   EXPECT_FLOAT_EQ(profile->ComputeStdDev(0), 0.5);
   EXPECT_FLOAT_EQ(profile->ComputeMean(1), 24.0);
   EXPECT_FLOAT_EQ(profile->ComputeStdDev(1), 1.0);
}

TEST(RProfileFillContext, StressFill)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t FlushEveryNFills = 500;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;

   // Fill a single bin, to maximize contention.
   auto profile = std::make_shared<RProfile>(1, std::make_pair(0, 1));
   {
      RProfileConcurrentFiller filler(profile);
      StressInParallel(NThreads, [&] {
         auto context = filler.CreateFillContext();
         for (std::size_t i = 0; i < NFillsPerThread; i++) {
            context->Fill(0.5, 1.5);
            if (i % FlushEveryNFills == 0) {
               context->Flush();
            }
         }
      });
   }

   EXPECT_EQ(profile->GetBinContent(0).fSum, NFills);
   EXPECT_EQ(profile->GetNEntries(), NFills);
   EXPECT_FLOAT_EQ(profile->ComputeNEffectiveEntries(), NFills);
   EXPECT_FLOAT_EQ(profile->ComputeMean(0), 0.5);
   EXPECT_FLOAT_EQ(profile->ComputeMean(1), 1.5);
}

TEST(RProfileFillContext, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   auto profile = std::make_shared<RProfile>(Bins, std::make_pair(0, Bins));

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      context->Fill(8.5, 23.0, RWeight(0.8));
      context->Fill(std::make_tuple(9.5), 25.0, RWeight(0.9));
   }

   auto &bin8 = profile->GetBinContent(RBinIndex(8));
   EXPECT_FLOAT_EQ(bin8.fSumValues, 18.4);
   EXPECT_FLOAT_EQ(bin8.fSumValues2, 423.2);
   EXPECT_FLOAT_EQ(bin8.fSum, 0.8);
   EXPECT_FLOAT_EQ(bin8.fSum2, 0.64);
   std::array<RBinIndex, 1> indices = {9};
   auto &bin9 = profile->GetBinContent(indices);
   EXPECT_FLOAT_EQ(bin9.fSumValues, 22.5);
   EXPECT_FLOAT_EQ(bin9.fSumValues2, 562.5);
   EXPECT_FLOAT_EQ(bin9.fSum, 0.9);
   EXPECT_FLOAT_EQ(bin9.fSum2, 0.81);

   EXPECT_EQ(profile->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile->GetStats().GetSumW(), 1.7);
   EXPECT_FLOAT_EQ(profile->GetStats().GetSumW2(), 1.45);
   // Cross-checked with TProfile
   EXPECT_FLOAT_EQ(profile->ComputeNEffectiveEntries(), 1.9931034);
   EXPECT_FLOAT_EQ(profile->ComputeMean(0), 9.0294118);
   EXPECT_FLOAT_EQ(profile->ComputeStdDev(0), 0.49913420);
   EXPECT_FLOAT_EQ(profile->ComputeMean(1), 24.058824);
   EXPECT_FLOAT_EQ(profile->ComputeStdDev(1), 0.99826840);
}

TEST(RProfileFillContext, StressFillWeight)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t FlushEveryNFills = 500;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;
   static constexpr double Weight = 0.5;

   // Fill a single bin, to maximize contention.
   auto profile = std::make_shared<RProfile>(1, std::make_pair(0, 1));
   {
      RProfileConcurrentFiller filler(profile);
      StressInParallel(NThreads, [&] {
         auto context = filler.CreateFillContext();
         for (std::size_t i = 0; i < NFillsPerThread; i++) {
            context->Fill(0.5, 1.5, RWeight(Weight));
            if (i % FlushEveryNFills == 0) {
               context->Flush();
            }
         }
      });
   }

   EXPECT_EQ(profile->GetBinContent(0).fSum, NFills * Weight);
   EXPECT_EQ(profile->GetNEntries(), NFills);
   EXPECT_FLOAT_EQ(profile->ComputeNEffectiveEntries(), NFills);
   EXPECT_FLOAT_EQ(profile->ComputeMean(0), 0.5);
   EXPECT_FLOAT_EQ(profile->ComputeMean(1), 1.5);
}

TEST(RProfileFillContext, FillCategorical)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   const std::vector<RAxisVariant> axes = {axis};
   auto profile = std::make_shared<RProfile>(axes);

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      context->Fill("b", 23.0);
      context->Fill(std::make_tuple("c"), 25.0);
   }

   auto &bin1 = profile->GetBinContent(RBinIndex(1));
   EXPECT_EQ(bin1.fSumValues, 23.0);
   EXPECT_EQ(bin1.fSumValues2, 529.0);
   EXPECT_EQ(bin1.fSum, 1.0);
   EXPECT_EQ(bin1.fSum2, 1.0);
   std::array<RBinIndex, 1> indices = {2};
   auto &bin2 = profile->GetBinContent(indices);
   EXPECT_EQ(bin2.fSumValues, 25.0);
   EXPECT_EQ(bin2.fSumValues2, 625.0);
   EXPECT_EQ(bin2.fSum, 1.0);
   EXPECT_EQ(bin2.fSum2, 1.0);

   EXPECT_EQ(profile->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile->ComputeNEffectiveEntries(), 2);
}

TEST(RProfileFillContext, FillCategoricalWeight)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   const std::vector<RAxisVariant> axes = {axis};
   auto profile = std::make_shared<RProfile>(axes);

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      context->Fill("b", 23.0, RWeight(0.8));
      context->Fill(std::make_tuple("c"), 25.0, RWeight(0.9));
   }

   auto &bin1 = profile->GetBinContent(RBinIndex(1));
   EXPECT_FLOAT_EQ(bin1.fSumValues, 18.4);
   EXPECT_FLOAT_EQ(bin1.fSumValues2, 423.2);
   EXPECT_FLOAT_EQ(bin1.fSum, 0.8);
   EXPECT_FLOAT_EQ(bin1.fSum2, 0.64);
   std::array<RBinIndex, 1> indices = {2};
   auto &bin2 = profile->GetBinContent(indices);
   EXPECT_FLOAT_EQ(bin2.fSumValues, 22.5);
   EXPECT_FLOAT_EQ(bin2.fSumValues2, 562.5);
   EXPECT_FLOAT_EQ(bin2.fSum, 0.9);
   EXPECT_FLOAT_EQ(bin2.fSum2, 0.81);

   EXPECT_EQ(profile->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(profile->GetStats().GetSumW(), 1.7);
   EXPECT_FLOAT_EQ(profile->GetStats().GetSumW2(), 1.45);
   // Cross-checked with TH1
   EXPECT_FLOAT_EQ(profile->ComputeNEffectiveEntries(), 1.9931034);
}

TEST(RProfileFillContext, FillForward)
{
   static constexpr std::size_t Bins = 20;
   auto profile = std::make_shared<RProfile>(Bins, std::make_pair(0, Bins));
   CopyArgument value(23.0);

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      std::tuple<CopyArgument> args(1.5);
      context->Fill(args, value);
      context->Fill(args, value, RWeight(0.5));
   }
   EXPECT_EQ(profile->GetNEntries(), 2);
   EXPECT_EQ(profile->GetBinContent(1).fSumValues, 34.5);

   ASSERT_FALSE(CopyArgument::HasBeenCopied());

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      CopyArgument arg(2.5);
      context->Fill(arg, value);
      context->Fill(arg, value, RWeight(0.5));
   }
   EXPECT_EQ(profile->GetNEntries(), 4);
   EXPECT_EQ(profile->GetBinContent(2).fSumValues, 34.5);

   ASSERT_FALSE(CopyArgument::HasBeenCopied());
}

TEST(RProfileFillContext, Flush)
{
   static constexpr std::size_t Bins = 20;
   auto profile = std::make_shared<RProfile>(Bins, std::make_pair(0, Bins));

   {
      RProfileConcurrentFiller filler(profile);
      auto context = filler.CreateFillContext();
      context->Fill(8.5, 23.0);
      // Flushing multiple times, explicitly and implicitly (in the destructor) should only add the entries once.
      context->Flush();
      context->Flush();
   }

   EXPECT_EQ(profile->GetNEntries(), 1);
   auto &bin8 = profile->GetBinContent(RBinIndex(8));
   EXPECT_EQ(bin8.fSumValues, 23.0);
   EXPECT_EQ(bin8.fSumValues2, 529.0);
   EXPECT_EQ(bin8.fSum, 1.0);
   EXPECT_EQ(bin8.fSum2, 1.0);
}
