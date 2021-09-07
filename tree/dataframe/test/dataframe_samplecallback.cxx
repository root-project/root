#define COUNTERHELPER_CALLBACKMODE
#include "CounterHelper.h"
#undef COUNTERHELPER_CALLBACKMODE

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RTrivialDS.hxx>
#include <TROOT.h> // ROOT::EnableImplicitMT
#include <TFile.h>
#include <TSystem.h>
#include <TTree.h>

#include <gtest/gtest.h>

#include <algorithm> // std::min
#include <memory>
#include <mutex>
#include <thread> // std::hardware_concurrency

// fixture for all tests in this file
struct RDFSampleCallback : ::testing::TestWithParam<bool> {
   unsigned int NSLOTS;
   ULong64_t NENTRIES = std::max(10u, std::thread::hardware_concurrency() * 2);

   RDFSampleCallback() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT();
   }

   ~RDFSampleCallback()
   {
      if (GetParam())
         ROOT::DisableImplicitMT();
   }
};

// A RAII object that ensures existence of nFiles root files named prefix0.root, prefix1.root, ...
// Each file contains a TTree called "t" with one `int` branch called "x" with sequentially increasing values (0,1,2...)
struct InputFilesRAII {
   unsigned int fNFiles = 0;
   std::string fPrefix;

   InputFilesRAII(unsigned int nFiles, std::string prefix) : fNFiles(nFiles), fPrefix(std::move(prefix))
   {
      for (auto i = 0u; i < fNFiles; ++i) {
         TFile f((fPrefix + std::to_string(i) + ".root").c_str(), "recreate");
         TTree t("t", "t");
         t.Branch("x", &i);
         t.Fill();
         t.Write();
      }
   }

   ~InputFilesRAII()
   {
      for (auto i = 0u; i < fNFiles; ++i)
         gSystem->Unlink((fPrefix + std::to_string(i) + ".root").c_str());
   }
};

TEST_P(RDFSampleCallback, EmptySource) {
   ROOT::RDataFrame df(NENTRIES);
   auto result = df.Book<>(CounterHelper(), {});
   // RDF with empty sources tries to produce 2 tasks per slot when MT is enabled
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, df.GetNSlots() * 2ull) : 1u;
   EXPECT_EQ(*result, expected);
}

TEST_P(RDFSampleCallback, DataSource) {
   auto df = ROOT::RDF::MakeTrivialDataFrame(NENTRIES);
   auto result = df.Book<>(CounterHelper(), {});
   // RTrivialDS tries to produce NSLOTS tasks
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, ULong64_t(df.GetNSlots())) : 1ull;
   EXPECT_EQ(*result, expected);
}

TEST_P(RDFSampleCallback, TTree) {
   const std::string prefix = "rdfdatablockcallback_ttree";
   InputFilesRAII file(1u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(CounterHelper(), {});
   EXPECT_EQ(*result, 1u);
}

TEST_P(RDFSampleCallback, TChain) {
   const std::string prefix = "rdfdatablockcallback_tchain";
   InputFilesRAII file(5u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(CounterHelper(), {});
   EXPECT_EQ(*result, 5u);
}

class SampleHelper : public ROOT::Detail::RDF::RActionImpl<SampleHelper> {
   std::shared_ptr<std::vector<ROOT::RDF::RSampleInfo>> fSamples;
   std::unique_ptr<std::mutex> fMutex;
public:
   SampleHelper()
      : fSamples(std::make_shared<std::vector<ROOT::RDF::RSampleInfo>>()), fMutex(std::make_unique<std::mutex>())
   {
   }
   SampleHelper(SampleHelper &&) = default;
   SampleHelper(const SampleHelper &) = delete;

   using Result_t = std::vector<ROOT::RDF::RSampleInfo>;
   auto GetResultPtr() const { return fSamples; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int) {}
   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int, const ROOT::RDF::RSampleInfo &db) mutable {
         std::lock_guard<std::mutex> lg(*fMutex);
         fSamples->emplace_back(db);
      };
   }
   void Finalize() {}

   std::string GetActionName() { return "SampleHelper"; }
};

TEST_P(RDFSampleCallback, EmptySourceSampleID) {
   ROOT::RDataFrame df(NENTRIES);
   auto result = df.Book<>(SampleHelper(), {});
   if (ROOT::IsImplicitMTEnabled()) {
      // RDF with empty sources tries to produce 2 tasks per slot when MT is enabled
      const auto expectedSize = std::min(NENTRIES, df.GetNSlots() * 2ull);
      ASSERT_EQ(result->size(), expectedSize);
      for (auto &id : *result) {
         // check that all entries start with the expected string
         EXPECT_TRUE(id.AsString().rfind("Empty source, range: {", 0) == 0);
         EXPECT_EQ(id.NEntries(), NENTRIES / expectedSize);
      }
   } else {
      ASSERT_EQ(result->size(), 1);
      const auto &id = result->at(0);
      const std::string expectedStr = "Empty source, range: {0, " + std::to_string(NENTRIES) + "}";
      EXPECT_EQ(id.AsString(), expectedStr);
      EXPECT_EQ(id.NEntries(), NENTRIES);
      EXPECT_EQ(id.EntryRange(), std::make_pair(0ull, NENTRIES));
   }
}

TEST_P(RDFSampleCallback, TTreeSampleID) {
   const std::string prefix = "rdfdatablockcallback_ttreedbnames";
   InputFilesRAII file(1u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(SampleHelper(), {});
   ASSERT_EQ(result->size(), 1u);
   const auto &id = result->at(0);
   EXPECT_TRUE(id.Contains(prefix));
   EXPECT_TRUE(id.AsString().rfind("/t") == id.AsString().size() - 2);
   EXPECT_EQ(id.NEntries(), 1u);
   EXPECT_EQ(id.EntryRange(), std::make_pair(0ull, 1ull));
}

TEST_P(RDFSampleCallback, TChainSampleID) {
   const std::string prefix = "rdfdatablockcallback_tchain";
   InputFilesRAII file(5u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(SampleHelper(), {});
   ASSERT_EQ(result->size(), 5u);
   std::vector<char> fileNumbers(5);
   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges(5);
   for (auto i = 0u; i < 5u; ++i) {
      EXPECT_TRUE(result->at(i).Contains(prefix));
      const auto &id = result->at(i).AsString();
      EXPECT_TRUE(id.rfind("/t") == id.size() - 2);
      fileNumbers[i] = id[id.size() - 8];
      entryRanges[i] = result->at(i).EntryRange();
   }

   std::sort(fileNumbers.begin(), fileNumbers.end());
   std::sort(
      entryRanges.begin(), entryRanges.end(),
      [](std::pair<ULong64_t, ULong64_t> p1, std::pair<ULong64_t, ULong64_t> p2) { return p1.first < p2.first; });
   for (int i = 0; i < 5; ++i) {
      // '0' == 48, '1' == 49, etc.
      EXPECT_EQ(fileNumbers[i], i + 48);
      // local entry numbers are reported
      EXPECT_EQ(entryRanges[i].first, 0ull);
      EXPECT_EQ(entryRanges[i].second, 1ull);
   }
}

/* TODO: data-block IDs for RDataSources are not supported yet
TEST_P(RDFSampleCallback, DataSource) {
   auto df = ROOT::RDF::MakeTrivialDataFrame(NENTRIES);
   auto result = df.Book<>(CounterHelper(), {});
   // RTrivialDS tries to produce NSLOTS tasks
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, df.GetNSlots()) : 1u;
   EXPECT_EQ(*result, expected);
}
*/

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFSampleCallback, ::testing::Values(false));

#ifdef R__USE_IMT
   // instantiate multi-thread tests
   INSTANTIATE_TEST_SUITE_P(MT, RDFSampleCallback, ::testing::Values(true));
#endif
