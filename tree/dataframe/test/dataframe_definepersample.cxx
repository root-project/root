#include <ROOT/RDataFrame.hxx>
#include <ROOT/RTrivialDS.hxx>
#include <TChain.h>
#include <TSystem.h>
#include <TTree.h>

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <thread> // std::thread::hardware_concurrency

// fixture for all tests in this file
struct DefinePerSample : ::testing::TestWithParam<bool> {
   unsigned int NSLOTS;
   unsigned int NENTRIES = std::max(10u, std::thread::hardware_concurrency() * 2);

   DefinePerSample() : NSLOTS(GetParam() ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam())
         ROOT::EnableImplicitMT();
   }

   ~DefinePerSample()
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
