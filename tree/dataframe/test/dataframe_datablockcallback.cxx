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
   const std::string prefix = "rdfdatablockcallback_ttree";
   InputFilesRAII file(5u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");
   auto result = df.Book<>(CounterHelper(), {});
   EXPECT_EQ(*result, 5u);
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, RDFDataBlockCallback, ::testing::Values(false));

#ifdef R__USE_IMT
   // instantiate multi-thread tests
   INSTANTIATE_TEST_SUITE_P(MT, RDFDataBlockCallback, ::testing::Values(true));
#endif
