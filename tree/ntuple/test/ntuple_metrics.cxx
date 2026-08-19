#include "TEnv.h"
#include "TKey.h"
#include "TSystem.h"
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

TEST(Metrics, IOMetrics)
{
   FileRaii fileGuard("test_ntuple_io_metrics.root");

   {
      auto model = RNTupleModel::Create();
      auto int_field = model->MakeField<int>("ints");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      for (int i = 0; i < 1000; ++i) {
         *int_field = i;
         ntuple->Fill();
      }
      ntuple->CommitCluster();
   }

   {
      auto ntupleReader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      EXPECT_FALSE(ntupleReader->GetMetrics().IsEnabled());
      ntupleReader->EnableMetrics();
      EXPECT_TRUE(ntupleReader->GetMetrics().IsEnabled());

      auto view = ntupleReader->GetView<int>("ints");
      for (auto i : *ntupleReader) {
         (void)view(i);
      }

      const auto &metrics = ntupleReader->GetMetrics();
      auto *randomness = metrics.GetCounter("RNTupleReader.RPageSourceFile.randomness");
      auto *sparseness = metrics.GetCounter("RNTupleReader.RPageSourceFile.sparseness");
      auto *szSkip = metrics.GetCounter("RNTupleReader.RPageSourceFile.szSkip");
      auto *szFile = metrics.GetCounter("RNTupleReader.RPageSourceFile.szFile");

      ASSERT_NE(randomness, nullptr);
      ASSERT_NE(sparseness, nullptr);
      ASSERT_NE(szSkip, nullptr);
      ASSERT_NE(szFile, nullptr);

      EXPECT_GE(szSkip->GetValueAsInt(), 0);
      EXPECT_GT(szFile->GetValueAsInt(), 0);
   }
}

namespace {

void WriteDummyEntries(std::string_view fileName, std::string_view ntupleName)
{
   auto model = RNTupleModel::Create();
   auto pt = model->MakeField<float>("float_field");

   auto writer = RNTupleWriter::Recreate(std::move(model), ntupleName, fileName);

   *pt = 1.0;
   writer->Fill();
   writer->CommitDataset();
}

void ReadMetrics(const std::string fileName, std::ostream &output)
{
   TFile file(fileName.c_str());

   std::set<std::string> ntupleNames;
   for (auto *keyObject : *file.GetListOfKeys()) {
      auto *key = static_cast<TKey *>(keyObject);
      if (std::string(key->GetClassName()) == "ROOT::RNTuple")
         ntupleNames.insert(key->GetName());
   }

   for (const auto &ntupleName : ntupleNames) {
      auto reader = ROOT::RNTupleReader::Open(ntupleName, fileName);
      const auto &descriptor = reader->GetDescriptor();

      output << ntupleName << "\n\n";
      for (auto counterId : descriptor.GetFieldZero().GetLinkIds()) {
         std::string fieldName = descriptor.GetFieldDescriptor(counterId).GetFieldName();
         std::string type = descriptor.GetFieldDescriptor(counterId).GetTypeName();

         output << std::string(8, ' ') << fieldName << " (" << type << ")\n";
      }
      output << "\n";
   }
}
} // namespace

TEST(Metrics, EnvironmentVariableExport)
{
   FileRaii fileGuard("test_environment_variable_export_export.root");

   const std::string metricsFileName = "environment_variable_export_metrics.root";

   gSystem->Setenv("ROOT_EXPERIMENTAL_EXPORT_RNTUPLE_METRICS", metricsFileName.c_str());

   WriteDummyEntries(fileGuard.GetPath().c_str(), "rntuple_name_1");
   WriteDummyEntries(fileGuard.GetPath().c_str(), "rntuple_name_1");
   WriteDummyEntries(fileGuard.GetPath().c_str(), "rntuple_name_2");

   gSystem->Unsetenv("ROOT_EXPERIMENTAL_EXPORT_RNTUPLE_METRICS");

   std::stringstream printedMetricsStream;
   ReadMetrics(metricsFileName, printedMetricsStream);

   const std::string printedMetricsString = std::move(printedMetricsStream).str();
   const std::string expected = R"(rntuple_name_1

        RNTupleWriter_RPageSinkBuf_ParallelZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeWallZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeWallCriticalSection (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeCpuZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeCpuCriticalSection (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_nPageCommitted (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_szWritePayload (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_szZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeWallWrite (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeWallZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeCpuWrite (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeCpuZip (std::int64_t)

rntuple_name_2

        RNTupleWriter_RPageSinkBuf_ParallelZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeWallZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeWallCriticalSection (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeCpuZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_timeCpuCriticalSection (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_nPageCommitted (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_szWritePayload (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_szZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeWallWrite (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeWallZip (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeCpuWrite (std::int64_t)
        RNTupleWriter_RPageSinkBuf_RPageSinkFile_timeCpuZip (std::int64_t)

)";

   EXPECT_EQ(printedMetricsString, expected);
}
