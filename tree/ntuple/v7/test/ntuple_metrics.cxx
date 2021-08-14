#include "ntuple_test.hxx"

#include <cmath>

TEST(Metrics, Counters)
{
   RNTupleMetrics metrics("test");
   EXPECT_FALSE(metrics.IsEnabled());

   RNTuplePlainCounter *ctrOne = nullptr;
   RNTupleAtomicCounter *ctrTwo = nullptr;
   ctrOne = metrics.MakeCounter<RNTuplePlainCounter *>("plain", "s", "example 1");
   ctrTwo = metrics.MakeCounter<RNTupleAtomicCounter *>("atomic", "s", "example 2");
   ASSERT_NE(nullptr, ctrOne);
   ASSERT_NE(nullptr, ctrTwo);
   EXPECT_FALSE(ctrOne->IsEnabled());
   EXPECT_FALSE(ctrTwo->IsEnabled());

   EXPECT_EQ(0, ctrOne->GetValue());
   ctrOne->Inc();
   ctrTwo->Inc();
   EXPECT_EQ(1, ctrOne->GetValue());
   EXPECT_EQ(0, ctrTwo->GetValue());
   metrics.Enable();
   EXPECT_TRUE(metrics.IsEnabled());
   ctrTwo->Inc();
   EXPECT_EQ(1, ctrTwo->XAdd(5));
   EXPECT_EQ(1, ctrOne->GetValue());
   EXPECT_EQ(6, ctrTwo->GetValue());

   RNTupleCalcPerf *ctrCalc = metrics.MakeCounter<RNTupleCalcPerf *>("calc", "s/s", "example 1/example2",
      metrics, [](const RNTupleMetrics &met) -> std::pair<bool, double> {
         auto ctr1 = met.GetCounter("test.plain");
         EXPECT_NE(ctr1, nullptr);
         auto ctr2 = met.GetCounter("test.atomic");
         EXPECT_NE(ctr2, nullptr);
         EXPECT_NE(ctr2->GetValueAsInt(), 0);
         return {true, (1.*ctr1->GetValueAsInt()) / ctr2->GetValueAsInt()};
      }
   );
   EXPECT_NE(ctrCalc, nullptr);
   EXPECT_DOUBLE_EQ(ctrCalc->GetValue(), 1./6.);
   EXPECT_NE(ctrCalc->ToString().find("calc"), std::string::npos);

   RNTupleCalcPerf *ctrCalcBad = metrics.MakeCounter<RNTupleCalcPerf *>("calcBad", "apples or oranges", "just bad",
      metrics, [](const RNTupleMetrics &) -> std::pair<bool, double> {
         return {false, 42.};
      }
   );
   EXPECT_NE(ctrCalcBad, nullptr);
   EXPECT_TRUE(std::isnan(ctrCalcBad->GetValue()));
   EXPECT_NE(ctrCalcBad->ToString(), ""); // whatever it is, it should not be empty or crash.
}

TEST(Metrics, Nested)
{
   RNTupleMetrics inner("inner");
   auto ctr = inner.MakeCounter<RNTuplePlainCounter *>("plain", "s", "example 1");

   RNTupleMetrics outer("outer");
   outer.ObserveMetrics(inner);

   outer.Enable();
   EXPECT_TRUE(ctr->IsEnabled());
   ctr->SetValue(42);

   EXPECT_EQ(nullptr, outer.GetCounter("a.b.c.d"));
   EXPECT_EQ(nullptr, outer.GetCounter("outer.xyz"));
   auto ctest = outer.GetCounter("outer.inner.plain");
   ASSERT_EQ(ctr, ctest);
   EXPECT_EQ(std::string("42"), ctest->GetValueAsString());
}

TEST(Metrics, Timer)
{
   RNTupleAtomicCounter ctrWallTime("wall time", "ns", "");
   ROOT::Experimental::Detail::RNTupleTickCounter<RNTupleAtomicCounter> ctrCpuTicks("cpu time", "ns", "");
   {
      RNTupleAtomicTimer timer(ctrWallTime, ctrCpuTicks);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
   }
   EXPECT_EQ(0U, ctrWallTime.GetValue());
   EXPECT_EQ(0U, ctrCpuTicks.GetValue());
   ctrWallTime.Enable();
   ctrCpuTicks.Enable();
   {
      RNTupleAtomicTimer timer(ctrWallTime, ctrCpuTicks);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
   }
   EXPECT_GT(ctrWallTime.GetValue(), 0U);
}

TEST(Metrics, RNTupleWriter)
{
   std::string rootFileName{"test_ntuple_writer_metrics.root"};
   FileRaii fileGuard(rootFileName);

   auto model = RNTupleModel::Create();
   auto int_field = model->MakeField<int>("ints");
   auto float_field = model->MakeField<float>("floats");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", rootFileName);
   EXPECT_FALSE(ntuple->GetMetrics().IsEnabled());
   ntuple->EnableMetrics();
   EXPECT_TRUE(ntuple->GetMetrics().IsEnabled());
   *int_field = 0;
   *float_field = 10.0;
   ntuple->Fill();
   ntuple->CommitCluster();
   auto* page_counter = ntuple->GetMetrics().GetCounter(
      "RNTupleWriter.RPageSinkBuf.RPageSinkFile.nPageCommitted");
   ASSERT_FALSE(page_counter == nullptr);
   // one page for the int field, one for the float field
   EXPECT_EQ(2, page_counter->GetValueAsInt());
}

TEST(Metrics, PresetIntervalHistogram)
{
   RNTupleMetrics inner("inner");
   std::vector<std::pair<uint64_t, uint64_t>> intervals = {
      std::make_pair(10, 20),
      std::make_pair(21, 30)
   };
   RNTupleHistoInterval counter("plain", "", "example 1", intervals);

   EXPECT_FALSE(inner.IsEnabled());
   inner.Enable();
   EXPECT_TRUE(inner.IsEnabled());

   counter.Fill(15);
   counter.Fill(9);

   // 14 and 20 in same interval as 15
   EXPECT_EQ(counter.GetBinContent(14), 1);
   EXPECT_EQ(counter.GetBinContent(20), 1);
   
   counter.Fill(20);
   EXPECT_EQ(counter.GetBinContent(19), 2);

   EXPECT_EQ(counter.GetBinContent(22), 0);

   counter.Fill(35);
   EXPECT_EQ(counter.GetBinContent(34), 0);
}

TEST(Metrics, LogHistogramUpperBound)
{
   RNTupleMetrics inner("inner");
   
   RNTupleHistoCounterLog counter("plain", "", "example 1", 1000);

   auto maxBound = counter.MaxLogUpperBound();

   // int(log2 of 1000) == 9 
   EXPECT_EQ(maxBound,9);
}

TEST(Metrics, LogHistogramCount) {
   RNTupleMetrics inner("inner");
   RNTupleHistoCounterLog counter("plain", "", "example 1", 1000);

   EXPECT_FALSE(inner.IsEnabled());
   inner.Enable();
   EXPECT_TRUE(inner.IsEnabled());

   counter.Fill(2);
   counter.Fill(3);
   counter.Fill(5);
   counter.Fill(6);
   counter.Fill(7);
   counter.Fill(8);

   // 2 entries with 1 exponent
   EXPECT_EQ(counter.GetExponentCount(1), 2);

   // 3 entries with 2 exponent
   EXPECT_EQ(counter.GetExponentCount(2), 3);

   // 1 entries with 8 exponent
   EXPECT_EQ(counter.GetExponentCount(3), 1);

   EXPECT_EQ(counter.GetOverflowCount(), 0);
   counter.Fill(1000);
   counter.Fill(1001);
   EXPECT_EQ(counter.GetOverflowCount(), 1);
}

TEST(Metrics, ActiveLearningHistogram) {
   RNTupleHistoActiveLearn counter("plain", "", "example 1", 10, 100);

   for(uint64_t i = 10; i < 110; i++) {
      counter.Fill(i);
      
      EXPECT_EQ(i, counter.GetMax());
   }

   // min fits bounds
   EXPECT_EQ(10, counter.GetMin());

   // max fits bounds
   EXPECT_EQ(109, counter.GetMax());

   // not yet flushed
   EXPECT_EQ(false, counter.IsFlushed());
   counter.Fill(109);

   // flush after 101th entry
   EXPECT_EQ(true, counter.IsFlushed());

   // 10 elems in range [70,79]
   EXPECT_EQ(10, counter.GetBinContent(77));

   // 12 bins created: 10 intervals + underflow + overflow
   EXPECT_EQ(10 + 2, counter.GetAll().size());

   // intervals match expected
   std::vector<std::pair<uint64_t, uint64_t>> intervals = {
      std::make_pair(0,9),
      std::make_pair(10, 19),
      std::make_pair(20, 29),
      std::make_pair(30, 39),
      std::make_pair(40, 49),
      std::make_pair(50, 59),
      std::make_pair(60, 69),
      std::make_pair(70, 79),
      std::make_pair(80, 89),
      std::make_pair(90, 99),
      std::make_pair(100, 109),
      std::make_pair(110, UINT64_MAX)
   };

   auto vcs = counter.GetAll();

   for(uint i = 1; i < vcs.size() - 1; i++) {
      EXPECT_EQ(intervals[i], vcs[i].first);
      EXPECT_EQ(10 + (i == 10), vcs[i].second);
   }

   // underflows are accounted for
   EXPECT_EQ(0, counter.GetUnderflow());
   for(uint i = 0; i < 10; i++) {
      counter.Fill(i);
   }
   EXPECT_EQ(10, counter.GetUnderflow());

   // overflows are accounted for
   EXPECT_EQ(0, counter.GetOverflow());
   for(uint i = 200; i < 220; i++) {
      counter.Fill(i);
   }
   EXPECT_EQ(20, counter.GetOverflow());
}

TEST(Metrics, FixedWidthIntervalHistogramZeroOffset) {
   RNTupleFixedWidthHistogram counter("a","","", 100, 199);

   counter.Fill(10);

   EXPECT_EQ(counter.GetBinContent(0), 0);
   EXPECT_EQ(counter.GetBinContent(49), 0);
   EXPECT_EQ(counter.GetBinContent(99), 0);
   EXPECT_EQ(counter.GetBinContent(100), 0);
   EXPECT_EQ(counter.GetBinContent(101), 0);
   EXPECT_EQ(counter.GetBinContent(199), 0);
   EXPECT_EQ(counter.GetBinContent(200), 0);
   EXPECT_EQ(counter.GetBinContent(201), 0);
   EXPECT_EQ(counter.GetBinContent(299), 0);
   EXPECT_EQ(counter.GetBinContent(300), 0);
   EXPECT_EQ(counter.GetBinContent(301), 0);
}