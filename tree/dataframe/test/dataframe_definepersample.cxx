#include <ROOT/RDataFrame.hxx>
#include <ROOT/RTrivialDS.hxx>
#include <TChain.h>
#include <TSystem.h>
#include <TTree.h>

#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <memory>
#include <thread> // std::thread::hardware_concurrency

#include <TFile.h>

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
   bool flag = false;
   auto df = ROOT::RDataFrame(1).DefinePerSample("x", [&flag](unsigned int, const ROOT::RDF::RSampleInfo &) {
      flag = true;
      return 0;
   });
   // Trigger the first execution of the event loop, the flag should be true.
   df.Count().GetValue();
   EXPECT_TRUE(flag);
   // Reset the flag and trigger again, flag should be again set to true after
   // the end of the second event loop.
   flag = false;
   df.Count().GetValue();
   EXPECT_TRUE(flag);
}

struct InputRNTuplesRAII {
   unsigned int fNFiles = 0;
   std::string fPrefix;

   InputRNTuplesRAII(unsigned int nFiles, std::string prefix) : fNFiles(nFiles), fPrefix(std::move(prefix))
   {
      for (auto i = 0u; i < fNFiles; ++i) {
         auto model = ROOT::RNTupleModel::Create();
         auto fldX = model->MakeField<int>("x");
         auto fn = fPrefix + std::to_string(i) + ".root";
         auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), "ntuple", fn);
         *fldX = i;
         ntpl->Fill();
      }
   }

   ~InputRNTuplesRAII()
   {
      for (auto i = 0u; i < fNFiles; ++i)
         std::remove((fPrefix + std::to_string(i) + ".root").c_str());
   }
};

TEST_P(DefinePerSample, RNTupleSingle)
{
   const std::string prefix = "rdfdefinepersample_rntuple";
   InputRNTuplesRAII file(1u, prefix);
   ROOT::RDataFrame df("ntuple", prefix + "0.root");

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
   const auto expected = 1u; // as the RNTuple only contains one cluster, we only have one "data-block"
   EXPECT_EQ(counter, expected);
}

TEST_P(DefinePerSample, RNTupleMany)
{
   const std::vector<std::string> fileNames{
      "rdfdefinepersample_rntuple_many0.root", "rdfdefinepersample_rntuple_many1.root",
      "rdfdefinepersample_rntuple_many2.root", "rdfdefinepersample_rntuple_many3.root",
      "rdfdefinepersample_rntuple_many4.root"};
   const std::string prefix{"rdfdefinepersample_rntuple_many"};
   InputRNTuplesRAII file(5u, prefix);
   ROOT::RDataFrame df("ntuple", fileNames);

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
   const auto expected = 5u; // one "data-block" per RNTuple (because each RNTuple only has one cluster)
   EXPECT_EQ(counter, expected);
}

struct DefinePerSampleWithDataset : ::testing::TestWithParam<std::pair<bool, bool>> {
   unsigned int fNSlots{};
   unsigned int fNEntries{5};
   std::string fDatasetName{"dataset"};
   std::vector<std::string> fFileNames{
      "rdf_definepersample_with_dataset_0.root", "rdf_definepersample_with_dataset_1.root",
      "rdf_definepersample_with_dataset_2.root", "rdf_definepersample_with_dataset_3.root",
      "rdf_definepersample_with_dataset_4.root"};

   void CreateRNTupleDataset()
   {
      for (const auto &fn : fFileNames) {
         auto model = ROOT::RNTupleModel::Create();
         auto fldX = model->MakeField<ULong64_t>("x");
         auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), fDatasetName, fn);
         for (ULong64_t entry = 0; entry < fNEntries; entry++) {
            *fldX = entry;
            ntpl->Fill();
            if (entry % 2 == 0)
               ntpl->CommitCluster();
         }
      }
   }

   void CreateTTreeDataset()
   {
      for (const auto &fn : fFileNames) {
         auto f = std::make_unique<TFile>(fn.c_str(), "recreate");
         auto t = std::make_unique<TTree>(fDatasetName.c_str(), fDatasetName.c_str());
         ULong64_t i{};
         t->Branch("x", &i);
         for (ULong64_t entry = 0; entry < fNEntries; entry++) {
            i = entry;
            t->Fill();
            if (entry % 2 == 0)
               t->FlushBaskets();
         }
         f->Write();
      }
   }

   DefinePerSampleWithDataset() : fNSlots(GetParam().first ? std::min(4u, std::thread::hardware_concurrency()) : 1u)
   {
      if (GetParam().first)
         ROOT::EnableImplicitMT();

      if (GetParam().second)
         CreateRNTupleDataset();
      else
         CreateTTreeDataset();
   }

   ~DefinePerSampleWithDataset() override
   {
      if (GetParam().first)
         ROOT::DisableImplicitMT();

      for (const auto &fn : fFileNames)
         std::remove(fn.c_str());
   }
};

template <typename T0, typename T1>
void expect_vec_eq(const T0 &v1, const T1 &v2)
{
   ASSERT_EQ(v1.size(), v2.size()) << "Vectors 'v1' and 'v2' are of unequal length";
   for (std::size_t i = 0ull; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "Vectors 'v1' and 'v2' differ at index " << i;
   }
}

TEST_P(DefinePerSampleWithDataset, CorrectSampleID)
{
   // Check that the sample id returned by the RSampleInfo is as expected
   ROOT::RDataFrame df{fDatasetName, fFileNames};
   auto df1 =
      df.DefinePerSample("sampleID", [](unsigned int, const ROOT::RDF::RSampleInfo &si) { return si.AsString(); });

   auto take = df1.Take<std::string>("sampleID");
   auto &sampleIDs = *take;
   // 5 files, 5 entries per file
   EXPECT_EQ(sampleIDs.size(), 25);
   // Sort to have consistent results also in MT case
   std::sort(sampleIDs.begin(), sampleIDs.end());
   std::vector<std::string> expectedIDs;
   expectedIDs.reserve(25);
   for (auto i = 0; i < 25; i++) {
      expectedIDs.push_back(fFileNames[i / 5] + '/' + fDatasetName);
   }
   expect_vec_eq(sampleIDs, expectedIDs);
}

std::string GetTestLabel(const testing::TestParamInfo<DefinePerSampleWithDataset::ParamType> &testInfo)
{
   if (testInfo.param.second)
      return "RNTuple";
   return "TTree";
}

// instantiate single-thread tests
INSTANTIATE_TEST_SUITE_P(Seq, DefinePerSample, ::testing::Values(false));
INSTANTIATE_TEST_SUITE_P(Seq, DefinePerSampleWithDataset,
                         ::testing::Values(std::make_pair(false, false), std::make_pair(false, true)), GetTestLabel);

#ifdef R__USE_IMT
// instantiate multi-thread tests
INSTANTIATE_TEST_SUITE_P(MT, DefinePerSample, ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(MT, DefinePerSampleWithDataset,
                         ::testing::Values(std::make_pair(true, false), std::make_pair(true, true)), GetTestLabel);
#endif
