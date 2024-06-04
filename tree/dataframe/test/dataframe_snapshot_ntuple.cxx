#include "ROOT/TestSupport.hxx"
#include "ROOT/RDataFrame.hxx"

#include "ROOT/RNTupleModel.hxx"
#include "ROOT/RNTupleWriter.hxx"
#include "ROOT/RNTupleReader.hxx"
#include "ROOT/RNTupleInspector.hxx" // For testing compression settings

#include "TROOT.h"
#include "TSystem.h"

#include "gtest/gtest.h"

#include "NTupleStruct.hxx"

using ROOT::Experimental::RNTupleInspector;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleReader;
using ROOT::Experimental::RNTupleWriter;

using namespace ROOT::RDF;

TEST(RDFSnapshotRNTuple, FromScratchTemplated)
{
   const auto filename = "RDFSnapshotRNTuple_from_scratch_templated.root";
   const std::vector<std::string> columns = {"x"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   auto sdf = df.Snapshot<int>("ntuple", filename, columns, opts);

   EXPECT_EQ(columns, sdf->GetColumnNames());

   auto ntuple = RNTupleReader::Open("ntuple", filename);
   EXPECT_EQ(25ull, ntuple->GetNEntries());

   auto x = ntuple->GetView<int>("x");
   for (const auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(10, x(i));
   }
}

TEST(RDFSnapshotRNTuple, FromScratchJITted)
{
   const auto filename = "RDFSnapshotRNTuple_from_scratch_jitted.root";
   const std::vector<std::string> columns = {"x"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   auto sdf = df.Snapshot("ntuple", filename, "x", opts);

   EXPECT_EQ(columns, sdf->GetColumnNames());

   auto ntuple = RNTupleReader::Open("ntuple", filename);
   EXPECT_EQ(25ull, ntuple->GetNEntries());

   auto x = ntuple->GetView<int>("x");
   for (const auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(10, x(i));
   }
}

void BookLazySnapshot()
{
   auto d = ROOT::RDataFrame(1);
   ROOT::RDF::RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
   opts.fLazy = true;
   d.Snapshot<ULong64_t>("t", "lazysnapshotnottriggered_shouldnotbecreated.root", {"rdfentry_"}, opts);
}

TEST(RDFSnapshotRNTuple, LazyNotTriggered)
{
   ROOT_EXPECT_WARNING(BookLazySnapshot(), "Snapshot", "A lazy Snapshot action was booked but never triggered.");
   // This returns FALSE if the file IS there.
   // TODO(fdegeus) use std::filesystem::exists once supported on all platforms.
   EXPECT_TRUE(gSystem->AccessPathName("lazysnapshotnottriggered_shouldnotbecreated.root"));
}

TEST(RDFSnapshotRNTuple, Compression)
{
   const auto filename = "RDFSnapshotRNTuple_compression.root";
   const std::vector<std::string> columns = {"x"};

   auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
   opts.fCompressionAlgorithm = ROOT::RCompressionSetting::EAlgorithm::kLZ4;
   opts.fCompressionLevel = 4;

   auto sdf = df.Snapshot("ntuple", filename, "x", opts);

   EXPECT_EQ(columns, sdf->GetColumnNames());

   auto inspector = RNTupleInspector::Create("ntuple", filename);
   EXPECT_EQ(404, inspector->GetCompressionSettings());
}

class RDFSnapshotRNTupleTest : public ::testing::Test {
protected:
   const std::string fFileName = "RDFSnapshotRNTuple.root";
   const std::string fNtplName = "ntuple";

   void SetUp() override
   {
      auto model = RNTupleModel::Create();
      *model->MakeField<float>("pt") = 42.f;
      *model->MakeField<std::string>("tag") = "xyz";
      auto fldNnlo = model->MakeField<std::vector<std::vector<float>>>("nnlo");
      fldNnlo->push_back(std::vector<float>());
      fldNnlo->push_back(std::vector<float>{1.0});
      fldNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
      *model->MakeField<ROOT::RVecI>("rvec") = ROOT::RVecI{1, 2, 3};
      auto fldElectron = model->MakeField<Electron>("electron");
      fldElectron->pt = 137.0;
      auto fldElectrons = model->MakeField<std::vector<Electron>>("electrons");
      fldElectrons->push_back(*fldElectron);
      fldElectrons->push_back(*fldElectron);
      auto fldJets = model->MakeField<std::vector<Jet>>("jets");
      fldJets->push_back(Jet{*fldElectrons});
      {
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNtplName, fFileName);
         ntuple->Fill();
      }
   }

   void TearDown() override { std::remove(fFileName.c_str()); }
};

TEST_F(RDFSnapshotRNTupleTest, DefaultToRNTupleTemplated)
{
   const auto filename = "RDFSnapshotRNTuple_snap_templated.root";

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot<float, int>("ntuple", filename, {"pt", "x"});

   auto ntuple = RNTupleReader::Open("ntuple", filename);
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   auto pt = ntuple->GetView<float>("pt");
   auto x = ntuple->GetView<int>("x");

   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_EQ(10, x(0));
}

TEST_F(RDFSnapshotRNTupleTest, DefaultToRNTupleJITted)
{
   const auto filename = "RDFSnapshotRNTuple_snap_jitted.root";

   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", filename, {"pt", "x"});

   auto ntuple = RNTupleReader::Open("ntuple", filename);
   EXPECT_EQ(1ull, ntuple->GetNEntries());

   auto pt = ntuple->GetView<float>("pt");
   auto x = ntuple->GetView<int>("x");

   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_EQ(10, x(0));
}

TEST_F(RDFSnapshotRNTupleTest, ToTTreeTemplated)
{
   const auto filename = "RDFSnapshotRNTuple_to_ttree_templated.root";

   auto df = ROOT::RDataFrame(fNtplName, fFileName);

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kTTree;

   auto sdf = df.Define("x", [] { return 10; }).Snapshot<float, int>("tree", filename, {"pt", "x"}, opts);

   TFile file(filename);
   auto tree = file.Get<TTree>("tree");
   EXPECT_EQ(1ull, tree->GetEntries());

   float pt;
   int x;

   tree->SetBranchAddress("pt", &pt);
   tree->SetBranchAddress("x", &x);

   tree->GetEntry(0);

   EXPECT_FLOAT_EQ(42.0, pt);
   EXPECT_EQ(10, x);
}

TEST_F(RDFSnapshotRNTupleTest, ToTTreeJITted)
{
   const auto filename = "RDFSnapshotRNTuple_to_ttree_jitted.root";

   auto df = ROOT::RDataFrame(fNtplName, fFileName);

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kTTree;

   auto sdf = df.Define("x", [] { return 10; }).Snapshot("tree", filename, {"pt", "x"}, opts);

   TFile file(filename);
   auto tree = file.Get<TTree>("tree");
   EXPECT_EQ(1ull, tree->GetEntries());

   float pt;
   int x;

   tree->SetBranchAddress("pt", &pt);
   tree->SetBranchAddress("x", &x);

   tree->GetEntry(0);

   EXPECT_FLOAT_EQ(42.0, pt);
   EXPECT_EQ(10, x);
}

TEST_F(RDFSnapshotRNTupleTest, ScalarFields)
{
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", "RDFSnapshotRNTuple_scalar_fields.root", "pt");

   std::vector<std::string> expected = {"pt"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto maxPt = sdf->Max<float>("pt").GetValue();
   EXPECT_FLOAT_EQ(42.f, maxPt);
}

TEST_F(RDFSnapshotRNTupleTest, VectorFields)
{
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", "RDFSnapshotRNTuple_all_fields.root", "nnlo");

   std::vector<std::string> expected = {"nnlo"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto nnloMax = sdf->Define("nnloMax",
                              [](const ROOT::RVec<ROOT::RVec<float>> &nnlo) {
                                 auto innerMax = ROOT::VecOps::Map(
                                    nnlo, [](ROOT::RVec<float> innerNnlo) { return ROOT::VecOps::Max(innerNnlo); });
                                 return ROOT::VecOps::Max(innerMax);
                              },
                              {"nnlo"})
                     .Max("nnloMax")
                     .GetValue();
   EXPECT_FLOAT_EQ(8.f, nnloMax);
}

TEST_F(RDFSnapshotRNTupleTest, ComplexFields)
{
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", "RDFSnapshotRNTuple_complex_fields.root", "electrons");

   std::vector<std::string> expected = {"electrons", "electrons.pt"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto electronsPtMax =
      sdf->Define("electronsPtMax", [](const ROOT::RVec<float> &electronPt) { return ROOT::VecOps::Max(electronPt); },
                  {"electrons.pt"})
         .Max("electronsPtMax")
         .GetValue();
   EXPECT_FLOAT_EQ(137.f, electronsPtMax);
}

TEST_F(RDFSnapshotRNTupleTest, InnerFields)
{
   auto df = ROOT::RDataFrame(fNtplName, fFileName);

   auto sdf1 = df.Snapshot("ntuple", "RDFSnapshotRNTuple_inner_fields.root", "electron.pt");

   std::vector<std::string> expected = {"electron_pt"};
   EXPECT_EQ(expected, sdf1->GetColumnNames());

   auto electronPtMax = sdf1->Max("electron_pt").GetValue();
   EXPECT_FLOAT_EQ(137.f, electronPtMax);

   auto sdf2 = df.Snapshot("ntuple", "RDFSnapshotRNTuple_inner_fields.root", "jets.electrons");

   expected = {"jets_electrons", "jets_electrons.pt"};
   EXPECT_EQ(expected, sdf2->GetColumnNames());

   auto jetsElectronsPtMax =
      sdf2
         ->Define("jetsElectronsPtMax",
                  [](const ROOT::RVec<ROOT::RVec<float>> &jetsElectronsPt) {
                     auto innerMax = ROOT::VecOps::Map(jetsElectronsPt, [](const ROOT::RVec<float> &electronsPt) {
                        return ROOT::VecOps::Max(electronsPt);
                     });
                     return ROOT::VecOps::Max(innerMax);
                  },
                  {"jets_electrons.pt"})
         .Max("jetsElectronsPtMax")
         .GetValue();
   EXPECT_FLOAT_EQ(137.f, jetsElectronsPtMax);
}

TEST_F(RDFSnapshotRNTupleTest, AllFields)
{
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Snapshot("ntuple", "RDFSnapshotRNTuple_all_fields.root");

   EXPECT_EQ(df.GetColumnNames(), sdf->GetColumnNames());
}

TEST_F(RDFSnapshotRNTupleTest, WithDefines)
{
   auto df = ROOT::RDataFrame(fNtplName, fFileName);
   auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", "RDFSnapshotRNTuple_with_defines.root");

   std::vector<std::string> expected = df.GetColumnNames();
   expected.push_back("x");
   EXPECT_EQ(expected, sdf->GetColumnNames());
}

TEST(RDFSnapshotRNTuple, WithFilters)
{
   const auto filename = "RDFSnapshotRNTuple_defines_and_filters.root";

   {
      auto df = ROOT::RDataFrame(10ull).DefineSlotEntry("x", [](unsigned int, std::uint64_t entry) { return entry; });

      RSnapshotOptions opts;
      opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

      df.Snapshot("ntuple", filename, "x", opts);
   }

   auto df = ROOT::RDataFrame("ntuple", filename).Filter("x % 2 == 0");
   auto sdf = df.Snapshot("ntuple", "snap_ntuple_filtered.root");
   auto ntuple = RNTupleReader::Open("ntuple", "snap_ntuple_filtered.root");
   EXPECT_EQ(5ull, ntuple->GetNEntries());

   auto x = ntuple->GetView<std::uint64_t>("x");
   for (const auto i : ntuple->GetEntryRange()) {
      EXPECT_FLOAT_EQ(i * 2, x(i));
   }
}

TEST(RDFSnapshotRNTuple, UpdateDifferentName)
{
   const auto filename = "RDFSnapshotRNTuple_update_different_name.root";

   {
      auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });
      RSnapshotOptions opts;
      opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
      auto sdf = df.Snapshot("ntuple", filename, "x", opts);
   }

   auto df = ROOT::RDataFrame("ntuple", filename);

   RSnapshotOptions opts;
   opts.fMode = "UPDATE";

   auto sdf = df.Define("y", [] { return 42; }).Snapshot("ntuple_snap", filename, "", opts);

   std::vector<std::string> expected = {"x", "y"};
   EXPECT_EQ(expected, sdf->GetColumnNames());

   auto ntupleOriginal = RNTupleReader::Open("ntuple", filename);
   EXPECT_EQ(25ull, ntupleOriginal->GetNEntries());

   auto ntupleSnap = RNTupleReader::Open("ntuple_snap", filename);
   EXPECT_EQ(25ull, ntupleSnap->GetNEntries());
}

TEST(RDFSnapshotRNTuple, UpdateSameName)
{
   const auto filename = "RDFSnapshotRNTuple_update_same_name.root";

   {
      auto df = ROOT::RDataFrame(25ull).Define("x", [] { return 10; });
      RSnapshotOptions opts;
      opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;
      auto sdf = df.Snapshot("ntuple", filename, "x", opts);
   }

   auto df = ROOT::RDataFrame("ntuple", filename);

   RSnapshotOptions opts;
   opts.fMode = "UPDATE";

   try {
      auto sdf = df.Define("y", [] { return 42; }).Snapshot<int, int>("ntuple", filename, {"x", "y"}, opts);
      FAIL() << "snapshotting in \"UPDATE\" mode to the same ntuple name without `fOverwriteIfExists` is not allowed ";
   } catch (const std::invalid_argument &err) {
      EXPECT_STREQ(err.what(),
                   "Snapshot: RNTuple \"ntuple\" already present in file "
                   "\"RDFSnapshotRNTuple_update_same_name.root\". If you want to delete the original "
                   "ntuple and write another, please set the 'fOverwriteIfExists' option to true in RSnapshotOptions.");
   }

   opts.fOverwriteIfExists = true;
   auto sdf = df.Define("y", [] { return 42; }).Snapshot("ntuple", filename, "", opts);

   std::vector<std::string> expected = {"x", "y"};
   EXPECT_EQ(expected, sdf->GetColumnNames());
}

void WriteTestTree(const std::string &tname, const std::string &fname)
{
   TFile file(fname.c_str(), "RECREATE");
   TTree t(tname.c_str(), tname.c_str());
   float pt;
   t.Branch("pt", &pt);

   pt = 42.0;
   t.Fill();

   t.Write();
}

TEST(RDFSnapshotRNTuple, DisallowFromTTreeTemplated)
{
   const auto treename = "tree";
   const auto filename = "RDFSnapshotRNTuple_disallow_from_ttree_templated.root";

   WriteTestTree(treename, filename);

   auto df = ROOT::RDataFrame(treename, filename);

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   try {
      auto sdf = df.Define("x", [] { return 10; }).Snapshot<float, int>("ntuple", filename, {"pt", "x"}, opts);
      FAIL() << "snapshotting from RNTuple to TTree is not (yet) possible";
   } catch (const std::runtime_error &err) {
      EXPECT_STREQ(err.what(), "Snapshotting from TTree to RNTuple is not yet supported. The current recommended way "
                               "to convert TTrees to RNTuple is through the RNTupleImporter.");
   }
}

TEST(RDFSnapshotRNTuple, DisallowFromTTreeJITted)
{
   const auto treename = "tree";
   const auto filename = "RDFSnapshotRNTuple_disallow_from_ttree_jitted.root";

   WriteTestTree(treename, filename);

   auto df = ROOT::RDataFrame(treename, filename);

   RSnapshotOptions opts;
   opts.fOutputFormat = ROOT::RDF::ESnapshotOutputFormat::kRNTuple;

   try {
      auto sdf = df.Define("x", [] { return 10; }).Snapshot("ntuple", filename, {"pt", "x"}, opts);
      FAIL() << "snapshotting from RNTuple to TTree is not (yet) possible";
   } catch (const std::runtime_error &err) {
      EXPECT_STREQ(err.what(), "Snapshotting from TTree to RNTuple is not yet supported. The current recommended way "
                               "to convert TTrees to RNTuple is through the RNTupleImporter.");
   }
}
