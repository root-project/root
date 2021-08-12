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
struct RDFDataBlockCallback : ::testing::TestWithParam<bool> {
   unsigned int NSLOTS;
   unsigned int NENTRIES = std::max(10u, std::thread::hardware_concurrency() * 2);

   RDFDataBlockCallback() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT();
   }

   ~RDFDataBlockCallback()
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

TEST_P(RDFDataBlockCallback, EmptySource) {
   ROOT::RDataFrame df(NENTRIES);
   auto result = df.Book<>(CounterHelper(), {});
   // RDF with empty sources tries to produce 2 tasks per slot when MT is enabled
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, df.GetNSlots() * 2u) : 1u;
   EXPECT_EQ(*result, expected);
}

TEST_P(RDFDataBlockCallback, DataSource) {
   auto df = ROOT::RDF::MakeTrivialDataFrame(NENTRIES);
   auto result = df.Book<>(CounterHelper(), {});
   // RTrivialDS tries to produce NSLOTS tasks
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, df.GetNSlots()) : 1u;
   EXPECT_EQ(*result, expected);
}

TEST_P(RDFDataBlockCallback, TTree) {
   const std::string prefix = "rdfdatablockcallback_ttree";
   InputFilesRAII file(1u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(CounterHelper(), {});
   EXPECT_EQ(*result, 1u);
}

TEST_P(RDFDataBlockCallback, TChain) {
   const std::string prefix = "rdfdatablockcallback_tchain";
   InputFilesRAII file(5u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(CounterHelper(), {});
   EXPECT_EQ(*result, 5u);
}

class DataBlockHelper : public ROOT::Detail::RDF::RActionImpl<DataBlockHelper> {
   std::shared_ptr<std::vector<ROOT::RDF::RDataBlockID>> fDataBlocks;
   std::unique_ptr<std::mutex> fMutex;
public:
   DataBlockHelper()
      : fDataBlocks(std::make_shared<std::vector<ROOT::RDF::RDataBlockID>>()), fMutex(std::make_unique<std::mutex>())
   {
   }
   DataBlockHelper(DataBlockHelper &&) = default;
   DataBlockHelper(const DataBlockHelper &) = delete;

   using Result_t = std::vector<ROOT::RDF::RDataBlockID>;
   auto GetResultPtr() const { return fDataBlocks; }
   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int) {}
   ROOT::RDF::DataBlockCallback_t GetDataBlockCallback() final
   {
      return [this](unsigned int, const ROOT::RDF::RDataBlockID &db) mutable {
         std::lock_guard<std::mutex> lg(*fMutex);
         fDataBlocks->emplace_back(db);
      };
   }
   void Finalize() {}

   std::string GetActionName() { return "DataBlockHelper"; }
};

TEST_P(RDFDataBlockCallback, EmptySourceDataBlockNames) {
   ROOT::RDataFrame df(NENTRIES);
   auto result = df.Book<>(DataBlockHelper(), {});
   if (ROOT::IsImplicitMTEnabled()) {
      // RDF with empty sources tries to produce 2 tasks per slot when MT is enabled
      const auto expectedSize = std::min(NENTRIES, df.GetNSlots() * 2u);
      ASSERT_EQ(result->size(), expectedSize);
      for (auto &id : *result) {
         // check that all entries start with the expected string
         EXPECT_TRUE(id.AsString().rfind("Empty source, range: {", 0) == 0);
      }
   } else {
      const std::string expectedStr = "Empty source, range: {0, " + std::to_string(NENTRIES) + "}";
      EXPECT_EQ(result->at(0).AsString(), expectedStr);
   }
}

TEST_P(RDFDataBlockCallback, TTreeDataBlockNames) {
   const std::string prefix = "rdfdatablockcallback_ttreedbnames";
   InputFilesRAII file(1u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(DataBlockHelper(), {});
   ASSERT_EQ(result->size(), 1u);
   EXPECT_TRUE(result->at(0).Contains(prefix));
   EXPECT_TRUE(result->at(0).AsString().rfind("/t") == result->at(0).AsString().size() - 2);
}

TEST_P(RDFDataBlockCallback, TChainDataBlockNames) {
   const std::string prefix = "rdfdatablockcallback_tchain";
   InputFilesRAII file(5u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(DataBlockHelper(), {});
   ASSERT_EQ(result->size(), 5u);
   std::vector<char> fileNumbers(5);
   for (auto i = 0u; i < 5u; ++i) {
      EXPECT_TRUE(result->at(i).Contains(prefix));
      const auto &id = result->at(i).AsString();
      EXPECT_TRUE(id.rfind("/t") == id.size() - 2);
      fileNumbers[i] = id[id.size() - 8];
   }

   std::sort(fileNumbers.begin(), fileNumbers.end());
   for (int i = 0; i < 5; ++i)
      // '0' == 48, '1' == 49, etc.
      EXPECT_EQ(fileNumbers[i], i + 48);
}

/* TODO: data-block IDs for RDataSources are not supported yet
TEST_P(RDFDataBlockCallback, DataSource) {
   auto df = ROOT::RDF::MakeTrivialDataFrame(NENTRIES);
   auto result = df.Book<>(CounterHelper(), {});
   // RTrivialDS tries to produce NSLOTS tasks
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, df.GetNSlots()) : 1u;
   EXPECT_EQ(*result, expected);
}
*/

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFDataBlockCallback, ::testing::Values(false));

#ifdef R__USE_IMT
   // instantiate multi-thread tests
   INSTANTIATE_TEST_SUITE_P(MT, RDFDataBlockCallback, ::testing::Values(true));
#endif
