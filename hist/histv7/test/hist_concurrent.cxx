#include "hist_test.hxx"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

TEST(RHistConcurrentFiller, Constructor)
{
   static constexpr std::size_t Bins = 20;
   auto hist = std::make_shared<RHist<int>>(Bins, std::make_pair(0, Bins));
   RHistConcurrentFiller filler(hist);

   std::shared_ptr<RHist<int>> histPtr = filler.GetHist();
   EXPECT_EQ(hist, histPtr);

   auto context = filler.CreateFillContext();
   context->Flush();

   EXPECT_THROW(RHistConcurrentFiller<int>(nullptr), std::invalid_argument);
}

TEST(RHistConcurrentFiller, OldEntries)
{
   static constexpr std::size_t Bins = 20;
   auto hist = std::make_shared<RHist<int>>(Bins, std::make_pair(0, Bins));
   hist->Fill(8.5);
   ASSERT_EQ(hist->GetNEntries(), 1);
   ASSERT_EQ(hist->GetBinContent(8), 1);

   {
      RHistConcurrentFiller filler(hist);
      auto context = filler.CreateFillContext();
      context->Flush();
   }

   EXPECT_EQ(hist->GetNEntries(), 1);
   EXPECT_EQ(hist->GetBinContent(8), 1);
}

TEST(RHistFillContext, Fill)
{
   static constexpr std::size_t Bins = 20;
   auto hist = std::make_shared<RHist<int>>(Bins, std::make_pair(0, Bins));

   {
      RHistConcurrentFiller filler(hist);
      auto context = filler.CreateFillContext();
      context->Fill(8.5);
      context->Fill(std::make_tuple(9.5));
   }

   EXPECT_EQ(hist->GetBinContent(RBinIndex(8)), 1);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_EQ(hist->GetBinContent(indices), 1);

   EXPECT_EQ(hist->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(hist->ComputeNEffectiveEntries(), 2);
   EXPECT_FLOAT_EQ(hist->ComputeMean(), 9);
   EXPECT_FLOAT_EQ(hist->ComputeStdDev(), 0.5);
}

TEST(RHistFillContext, StressFill)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t FlushEveryNFills = 500;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;

   // Fill a single bin, to maximize contention.
   auto hist = std::make_shared<RHist<int>>(1, std::make_pair(0, 1));
   {
      RHistConcurrentFiller filler(hist);
      StressInParallel(NThreads, [&] {
         auto context = filler.CreateFillContext();
         for (std::size_t i = 0; i < NFillsPerThread; i++) {
            context->Fill(0.5);
            if (i % FlushEveryNFills == 0) {
               context->Flush();
            }
         }
      });
   }

   EXPECT_EQ(hist->GetBinContent(0), NFills);
   EXPECT_EQ(hist->GetNEntries(), NFills);
   EXPECT_FLOAT_EQ(hist->ComputeNEffectiveEntries(), NFills);
   EXPECT_FLOAT_EQ(hist->ComputeMean(), 0.5);
}

TEST(RHistFillContext, FillWeight)
{
   static constexpr std::size_t Bins = 20;
   auto hist = std::make_shared<RHist<float>>(Bins, std::make_pair(0, Bins));

   {
      RHistConcurrentFiller filler(hist);
      auto context = filler.CreateFillContext();
      context->Fill(8.5, RWeight(0.8));
      context->Fill(std::make_tuple(9.5), RWeight(0.9));
   }

   EXPECT_FLOAT_EQ(hist->GetBinContent(RBinIndex(8)), 0.8);
   std::array<RBinIndex, 1> indices = {9};
   EXPECT_FLOAT_EQ(hist->GetBinContent(indices), 0.9);

   EXPECT_EQ(hist->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(hist->GetStats().GetSumW(), 1.7);
   EXPECT_FLOAT_EQ(hist->GetStats().GetSumW2(), 1.45);
   // Cross-checked with TH1
   EXPECT_FLOAT_EQ(hist->ComputeNEffectiveEntries(), 1.9931034);
   EXPECT_FLOAT_EQ(hist->ComputeMean(), 9.0294118);
   EXPECT_FLOAT_EQ(hist->ComputeStdDev(), 0.49913420);
}

TEST(RHistFillContext, StressFillWeight)
{
   static constexpr std::size_t NThreads = 4;
   static constexpr std::size_t NFillsPerThread = 10000;
   static constexpr std::size_t FlushEveryNFills = 500;
   static constexpr std::size_t NFills = NThreads * NFillsPerThread;
   static constexpr double Weight = 0.5;

   // Fill a single bin, to maximize contention.
   auto hist = std::make_shared<RHist<float>>(1, std::make_pair(0, 1));
   {
      RHistConcurrentFiller filler(hist);
      StressInParallel(NThreads, [&] {
         auto context = filler.CreateFillContext();
         for (std::size_t i = 0; i < NFillsPerThread; i++) {
            context->Fill(0.5, RWeight(Weight));
            if (i % FlushEveryNFills == 0) {
               context->Flush();
            }
         }
      });
   }

   EXPECT_EQ(hist->GetBinContent(0), NFills * Weight);
   EXPECT_EQ(hist->GetNEntries(), NFills);
   EXPECT_FLOAT_EQ(hist->ComputeNEffectiveEntries(), NFills);
   EXPECT_FLOAT_EQ(hist->ComputeMean(), 0.5);
}

TEST(RHistFillContext, FillCategorical)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   const std::vector<RAxisVariant> axes = {axis};
   auto hist = std::make_shared<RHist<int>>(axes);

   {
      RHistConcurrentFiller filler(hist);
      auto context = filler.CreateFillContext();
      context->Fill("b");
      context->Fill(std::make_tuple("c"));
   }

   EXPECT_EQ(hist->GetBinContent(RBinIndex(1)), 1);
   std::array<RBinIndex, 1> indices = {2};
   EXPECT_EQ(hist->GetBinContent(indices), 1);

   EXPECT_EQ(hist->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(hist->ComputeNEffectiveEntries(), 2);
}

TEST(RHistFillContext, FillCategoricalWeight)
{
   const std::vector<std::string> categories = {"a", "b", "c"};
   const RCategoricalAxis axis(categories);
   const std::vector<RAxisVariant> axes = {axis};
   auto hist = std::make_shared<RHist<float>>(axes);

   {
      RHistConcurrentFiller filler(hist);
      auto context = filler.CreateFillContext();
      context->Fill("b", RWeight(0.8));
      context->Fill(std::make_tuple("c"), RWeight(0.9));
   }

   EXPECT_FLOAT_EQ(hist->GetBinContent(RBinIndex(1)), 0.8);
   std::array<RBinIndex, 1> indices = {2};
   EXPECT_FLOAT_EQ(hist->GetBinContent(indices), 0.9);

   EXPECT_EQ(hist->GetNEntries(), 2);
   EXPECT_FLOAT_EQ(hist->GetStats().GetSumW(), 1.7);
   EXPECT_FLOAT_EQ(hist->GetStats().GetSumW2(), 1.45);
   // Cross-checked with TH1
   EXPECT_FLOAT_EQ(hist->ComputeNEffectiveEntries(), 1.9931034);
}

TEST(RHistFillContext, Flush)
{
   static constexpr std::size_t Bins = 20;
   auto hist = std::make_shared<RHist<int>>(Bins, std::make_pair(0, Bins));

   {
      RHistConcurrentFiller filler(hist);
      auto context = filler.CreateFillContext();
      context->Fill(8.5);
      // Flushing multiple times, explicitly and implicitly (in the destructor) should only add the entries once.
      context->Flush();
      context->Flush();
   }

   EXPECT_EQ(hist->GetNEntries(), 1);
   EXPECT_EQ(hist->GetBinContent(RBinIndex(8)), 1);
}
