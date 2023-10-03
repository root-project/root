#include <ROOT/RDataFrame.hxx>
#include <ROOT/RTrivialDS.hxx>
#include <TChain.h>
#include <TSystem.h>
#include <TTree.h>

#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

#include <atomic>
#include <memory>
#include <thread> // std::thread::hardware_concurrency
#include <vector>

template <typename T>
void EXPECT_VEC_EQ(const std::vector<T> &v1, const std::vector<T> &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

// fixture for all tests in this file
struct DefinePerSample : ::testing::TestWithParam<bool> {
   unsigned int NSLOTS;
   unsigned int NENTRIES = std::max(10u, std::thread::hardware_concurrency() * 2);

   DefinePerSample() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT();
   }

   ~DefinePerSample() override
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

TEST_P(DefinePerSample, NoJitting)
{
   std::atomic_int counter{0};
   auto df = ROOT::RDataFrame(NENTRIES).DefinePerSample("x", [&counter](unsigned int, const ROOT::RDF::RSampleInfo &) {
      ++counter;
      return 42;
   });
   auto xmin = df.Min<int>("x");
   auto xmax = df.Max<int>("x");
   EXPECT_EQ(*xmin, 42);
   EXPECT_EQ(*xmax, 42);
   // RDF with empty sources tries to produce 2 tasks per slot when MT is enabled
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(NENTRIES, df.GetNSlots() * 2u) : 1u;
   EXPECT_EQ(counter, expected);
}

int AtomicIntValueFromInterpreter(std::string_view varName)
{
   return int(*reinterpret_cast<std::atomic_int *>(gInterpreter->Calc(varName.data())));
}

TEST(DefinePerSample, Jitted)
{
   gInterpreter->Declare("std::atomic_int rdftestcounter1{0};");
   auto df = ROOT::RDataFrame(3).DefinePerSample("x", "rdftestcounter1++; return 42;");
   auto xmin = df.Min<int>("x");
   auto xmax = df.Max<int>("x");
   EXPECT_EQ(*xmin, 42);
   EXPECT_EQ(*xmax, 42);
   // RDF with empty sources tries to produce 2 tasks per slot when MT is enabled
   const auto expected = ROOT::IsImplicitMTEnabled() ? std::min(3u, df.GetNSlots() * 2u) : 1u;
   EXPECT_EQ(AtomicIntValueFromInterpreter("rdftestcounter1"), expected);
}

TEST_P(DefinePerSample, Tree)
{
   const std::string prefix = "rdfdefinepersample_tree";
   InputFilesRAII file(1u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");

   std::atomic_int counter{0};
   auto df2 = df.DefinePerSample("y", [&counter](unsigned int, const ROOT::RDF::RSampleInfo &db) {
      EXPECT_EQ(db.EntryRange(), std::make_pair(0ull, 1ull));
      ++counter;
      return 42;
   });
   auto xmin = df2.Min<int>("y");
   auto xmax = df2.Max<int>("y");
   EXPECT_EQ(*xmin, 42);
   EXPECT_EQ(*xmax, 42);
   const auto expected = 1u; // as the TTree only contains one cluster, we only have one "data-block"
   EXPECT_EQ(counter, expected);
}

TEST_P(DefinePerSample, TChain)
{
   const std::string prefix = "rdfdefinepersample_chain";
   InputFilesRAII file(5u, prefix);
   ROOT::RDataFrame df("t", prefix + "*");

   std::atomic_int counter{0};
   auto df2 = df.DefinePerSample("y", [&counter](unsigned int, const ROOT::RDF::RSampleInfo &db) {
      EXPECT_EQ(db.EntryRange(), std::make_pair(0ull, 1ull));
      ++counter;
      return 42;
   });
   auto xmin = df2.Min<int>("y");
   auto xmax = df2.Max<int>("y");
   EXPECT_EQ(*xmin, 42);
   EXPECT_EQ(*xmax, 42);
   const auto expected = 5u; // one "data-block" per tree (because each tree only has one cluster)
   EXPECT_EQ(counter, expected);
}

TEST(DefinePerSampleMore, ThrowOnRedefinition)
{
   auto df = ROOT::RDataFrame(1)
                .Define("x", [] { return 42; });
   EXPECT_THROW(df.DefinePerSample("x", [](unsigned, const ROOT::RDF::RSampleInfo &) { return 42; }),
                std::runtime_error);
}

TEST(DefinePerSampleMore, GetColumnType)
{
   auto df = ROOT::RDataFrame(1).DefinePerSample("x", [](unsigned, const ROOT::RDF::RSampleInfo &) { return 42; });
   EXPECT_EQ(df.GetColumnType("x"), "int");
}

TEST(DefinePerSampleMore, GetColumnNames)
{
   auto df = ROOT::RDataFrame(1).DefinePerSample("x", [](unsigned, const ROOT::RDF::RSampleInfo &) { return 42; });
   EXPECT_EQ(df.GetColumnNames(), std::vector<std::string>{"x"});
}

TEST(DefinePerSampleMore, GetDefinedColumnNames)
{
   auto df = ROOT::RDataFrame(1).DefinePerSample("x", [](unsigned, const ROOT::RDF::RSampleInfo &) { return 42; });
   EXPECT_EQ(df.GetDefinedColumnNames(), std::vector<std::string>{"x"});
}

// Regression test for https://github.com/root-project/root/issues/12043
TEST(DefinePerSample, TwoExecutions)
{
   unsigned int nFiles{5};
   std::string prefix{"definepersample_twoexecutions"};
   std::vector<std::string> fileNames(nFiles);
   std::generate(fileNames.begin(), fileNames.end(), [n = 0, &prefix]() mutable {
      auto name = prefix + std::to_string(n) + ".root";
      n++;
      return name;
   });
   InputFilesRAII files{nFiles, prefix};
   std::vector<double> weights(nFiles);
   std::iota(weights.begin(), weights.end(), 1.);

   ROOT::RDataFrame df{"t", fileNames};

   auto weightsCol =
      df.DefinePerSample("weightbyfile", [&fileNames, &weights](unsigned int, const ROOT::RDF::RSampleInfo &id) {
         auto thisFileName = id.AsString();
         // The ID of the sample is "filename/treename". Erase "/t" from the
         // string to get only the file name
         thisFileName.erase(thisFileName.find('/'), 2);
         std::size_t thisFileIdx =
            std::distance(fileNames.begin(), std::find(fileNames.begin(), fileNames.end(), thisFileName));
         return thisFileIdx == fileNames.size() ? -1. : weights[thisFileIdx];
      });

   // Trigger two executions in a row of the same action.
   auto t0 = weightsCol.Take<double>("weightbyfile");
   auto &t0Vals = *t0;
   // Sort values for the MT test
   std::sort(t0Vals.begin(), t0Vals.end());
   EXPECT_VEC_EQ(*t0, {1, 2, 3, 4, 5});
   auto t1 = weightsCol.Take<double>("weightbyfile");
   auto &t1Vals = *t0;
   // Sort values for the MT test
   std::sort(t1Vals.begin(), t1Vals.end());
   EXPECT_VEC_EQ(*t1, {1, 2, 3, 4, 5});
}

/* TODO
// Not supported yet
TEST(DefinePerSample, DataSource)
{
   ROOT::RDataFrame df(std::make_unique<ROOT::RDF::RTrivialDS>(1));
   auto r = df.DefinePerSample("col0", [] { return 42; }).Max<int>("col0");
   EXPECT_EQ(*r, 42);
}
*/

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, DefinePerSample, ::testing::Values(false));

#ifdef R__USE_IMT
// instantiate multi-thread tests
INSTANTIATE_TEST_SUITE_P(MT, DefinePerSample, ::testing::Values(true));
#endif
